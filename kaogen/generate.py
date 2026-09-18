"""Generate kaomojis from a natural-language prompt (FR or EN).

    python -m kaogen.generate "un chat timide qui se cache" -n 4
"""
import argparse

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .train_lm import PROMPT_FMT, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=4, help="number of candidates")
    ap.add_argument("--adapter", default="runs/lm")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    args = ap.parse_args()

    device = pick_device()
    base = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()

    enc = tok(PROMPT_FMT.format(prompt=args.prompt.strip().lower()), return_tensors="pt",
              add_special_tokens=False).to(device)
    with torch.no_grad():
        # no repetition penalty: kaomojis repeat characters on purpose (━━━, ｡｡｡)
        out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                             num_return_sequences=args.n, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    for i, seq in enumerate(out, 1):
        print(f"--- {i}\n{tok.decode(seq[enc['input_ids'].shape[1]:], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
