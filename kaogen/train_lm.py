"""LoRA fine-tune a small *base* LM on prompt -> kaomoji pairs.

A base (non-chat) model in a fixed non-chat format has no refusal behaviour to trigger:
it only ever learns to continue "### Kaomoji:" with a drawing.

    python -m kaogen.train_lm --max-steps 3000
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_FMT = "### Prompt: {prompt}\n### Kaomoji:\n"
SAMPLE_PROMPTS = ["angry cat", "un ours qui fait un câlin", "table flip", "big multi-line bunny with a heart"]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode(tok, row: dict, max_len: int):
    prompt = tok(PROMPT_FMT.format(prompt=row["prompt"]), add_special_tokens=False)["input_ids"]
    target = tok(row["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(prompt) + len(target) > max_len:
        return None  # skip, never truncate: a cut-off drawing teaches the model to draw cut-off art
    return prompt + target, [-100] * len(prompt) + target


def load_split(path: Path, tok, max_len: int) -> list:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    examples = [e for e in (encode(tok, r, max_len) for r in rows) if e is not None]
    print(f"{path}: {len(examples)}/{len(rows)} examples fit in {max_len} tokens "
          f"({sum(len(x) for x, _ in examples) / 1e6:.2f}M tokens per epoch)", flush=True)
    return examples


def batches(examples: list, batch_tokens: int, pad_id: int, rng: random.Random | None):
    """Yield padded batches holding at most `batch_tokens` padded tokens.

    A token budget (not an example count) bounds memory: with a 152k vocabulary the logits dominate,
    so 8 multi-line drawings cost ~10x more than 8 one-liners. Short kaomojis get fuller batches.
    """
    order = list(range(len(examples)))
    if rng:
        rng.shuffle(order)
    out = []
    for i in range(0, len(order), 2048):
        bucket = sorted(order[i : i + 2048], key=lambda j: len(examples[j][0]))
        cur = []
        for j in bucket:
            # ascending lengths: the newcomer is the longest, so padded size = count * its length
            if cur and (len(cur) + 1) * len(examples[j][0]) > batch_tokens:
                out.append(cur)
                cur = []
            cur.append(j)
        if cur:
            out.append(cur)
    if rng:
        rng.shuffle(out)
    for idx in out:
        width = max(len(examples[j][0]) for j in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        labels = torch.full((len(idx), width), -100, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, j in enumerate(idx):
            x, y = examples[j]
            ids[row, : len(x)] = torch.tensor(x)
            labels[row, : len(y)] = torch.tensor(y)
            mask[row, : len(x)] = 1
        yield ids, labels, mask


def release_cache(device: str) -> None:
    # Token-budget batches have ever-changing shapes and the MPS allocator caches blocks for each
    # one: measured 6.4 GB held for 3.6 GB live, then OOM. Emptying it costs no throughput.
    if device == "mps":
        torch.mps.empty_cache()


def lm_loss(model, ids, mask, labels) -> torch.Tensor:
    """Cross-entropy on target positions only.

    `model(labels=...)` builds [batch, seq, 152k] logits, mostly for prompt and padding positions
    that the loss ignores; that tensor dominates memory. Project only the hidden states we score.
    """
    base = model.get_base_model()  # LoRA layers are injected in place, so this still trains them
    hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    keep = targets != -100
    if hidden.device.type == "mps" and hidden.dtype != torch.float32:
        # MPS has no bfloat16 backward for boolean-mask indexing; CUDA does, and the float32
        # round trip there would cost a full copy of the hidden states for nothing.
        selected = hidden.float()[keep].to(hidden.dtype)
    else:
        selected = hidden[keep]
    return torch.nn.functional.cross_entropy(base.lm_head(selected).float(), targets[keep])


@torch.no_grad()
def evaluate(model, val, args, pad_id: int, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (ids, labels, mask) in enumerate(batches(val, args.batch_tokens, pad_id, None)):
        if i >= args.eval_batches:
            break
        total += lm_loss(model, ids.to(device), mask.to(device), labels.to(device)).item()
        n += 1
        release_cache(device)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def sample(model, tok, prompt: str, device: str, max_new_tokens: int = 200) -> str:
    model.eval()
    enc = tok(PROMPT_FMT.format(prompt=prompt), return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model.train()
    release_cache(device)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="runs/lm")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-tokens", type=int, default=0,
                    help="padded tokens per micro-batch (memory bound); 0 picks a per-device default")
    ap.add_argument("--grad-accum", type=int, default=0, help="0 picks a per-device default")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in backward: 2.3 GB -> 0.3 GB at 512 tokens, 25-45%% slower")
    ap.add_argument("--mps-memory-fraction", type=float, default=0.6,
                    help="cap on MPS memory: past it PyTorch raises OOM instead of letting macOS swap")
    ap.add_argument("--max-steps", type=int, default=3000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng, device = random.Random(args.seed), pick_device()
    if device == "mps":
        # On a 16 GB Mac shared with an IDE, an over-large batch does not fail: it swaps for hours.
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    # A discrete GPU has far more headroom and is starved by Mac-sized micro-batches; keeping the
    # product batch_tokens * grad_accum similar means the two devices train the same recipe.
    if not args.batch_tokens:
        args.batch_tokens = 8192 if device == "cuda" else 512
    if not args.grad_accum:
        args.grad_accum = 1 if device == "cuda" else 4
    print(f"device {device} | batch-tokens {args.batch_tokens} | grad-accum {args.grad_accum}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train = load_split(Path("data/train.jsonl"), tok, args.max_len)
    val = load_split(Path("data/val.jsonl"), tok, args.max_len)

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    step, micro, tokens, running, t0 = 0, 0, 0, [], time.time()
    out_dir = Path(args.out)
    while step < args.max_steps:
        for ids, labels, mask in batches(train, args.batch_tokens, tok.pad_token_id, rng):
            loss = lm_loss(model, ids.to(device), mask.to(device), labels.to(device))
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            del loss
            release_cache(device)
            tokens += int(mask.sum())
            micro += 1
            if micro % args.grad_accum:
                continue
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                mem = f" | {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if device == "mps" else ""
                print(f"step {step:5d} | loss {sum(running) / len(running):.4f} | lr {lr_at(step):.2e} | "
                      f"{tokens / (time.time() - t0):.0f} tok/s{mem}", flush=True)
                running = []
            if step % args.eval_every == 0 or step == args.max_steps:
                print(f"--- step {step}: val loss {evaluate(model, val, args, tok.pad_token_id, device):.4f}")
                for p in SAMPLE_PROMPTS:
                    print(f"[{p}]\n{sample(model, tok, p, device)}\n")
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
            if step >= args.max_steps:
                break
    print(f"done: adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
