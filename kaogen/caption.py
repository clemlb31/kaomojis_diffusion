"""Caption kaomojis in EN + FR with Claude through the Message Batches API (50% cheaper).

    python -m kaogen.caption submit      # send everything not captioned yet
    python -m kaogen.caption fetch       # poll, then append results to data/captions.jsonl

NSFW items are never sent: the API may decline them, and one refusal would void the
whole group. They get tag-based prompts in build_pairs.py instead.
"""
import argparse
import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DATASET = Path("data/kaomojis.jsonl")
CAPTIONS = Path("data/captions.jsonl")
STATE = Path("data/caption_batches.json")
GROUP_SIZE = 8
MAX_REQUESTS_PER_BATCH = 10_000

SYSTEM = """You write search-style descriptions for kaomojis and multi-line Unicode text art, \
to train a text-to-kaomoji generator.

For each item you receive its id, its community tags (noisy, may be wrong) and the art itself. \
Look at the art: trust what you see over the tags.

For every item write 3 English and 3 French descriptions of what a user would type to get it:
- one terse keyword-style query ("angry cat", "chat en colère")
- one natural sentence describing subject, emotion and action
- one request phrased as an ask ("draw me a ...", "fais-moi un ...")
Mention size when it matters (tiny one-liner vs. big multi-line drawing). Lowercase, no quotes, \
never copy characters from the art, 3 to 15 words each. French must be natural French, \
not word-for-word translation."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "en": {"type": "array", "items": {"type": "string"}},
                    "fr": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "en", "fr"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def render(group: list[dict]) -> str:
    parts = []
    for rec in group:
        tags = ", ".join(rec["tags"][:12]) or "(none)"
        parts.append(f"<item id=\"{rec['id']}\">\n<tags>{tags}</tags>\n<art>\n{rec['text']}\n</art>\n</item>")
    return "\n\n".join(parts)


def submit(args) -> None:
    done = {c["id"] for c in load_jsonl(CAPTIONS)}
    todo = [r for r in load_jsonl(DATASET) if not r["nsfw"] and r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to caption")
        return
    groups = [todo[i : i + GROUP_SIZE] for i in range(0, len(todo), GROUP_SIZE)]
    print(f"{len(todo)} kaomojis -> {len(groups)} requests, model {args.model}")

    client = anthropic.Anthropic()
    state = json.loads(STATE.read_text()) if STATE.exists() else []
    for start in range(0, len(groups), MAX_REQUESTS_PER_BATCH):
        requests = [
            Request(
                custom_id=f"g{start + i}",
                params=MessageCreateParamsNonStreaming(
                    model=args.model,
                    max_tokens=4000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": render(group)}],
                    output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                ),
            )
            for i, group in enumerate(groups[start : start + MAX_REQUESTS_PER_BATCH])
        ]
        try:
            batch = client.messages.batches.create(requests=requests)
        except anthropic.AuthenticationError:
            raise SystemExit("no valid credentials: export ANTHROPIC_API_KEY (or run `ant auth login`)")
        except anthropic.RateLimitError as e:
            raise SystemExit(f"rate limited, retry in {e.response.headers.get('retry-after', '60')}s")
        except anthropic.APIStatusError as e:
            raise SystemExit(f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            raise SystemExit(f"network error: {e}")
        state.append({"id": batch.id, "fetched": False})
        STATE.write_text(json.dumps(state, indent=2))
        print(f"submitted {batch.id} ({len(requests)} requests)")


def fetch(args) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else []
    client = anthropic.Anthropic()
    valid_ids = {r["id"] for r in load_jsonl(DATASET)}
    for entry in state:
        if entry["fetched"]:
            continue
        while True:
            batch = client.messages.batches.retrieve(entry["id"])
            if batch.processing_status == "ended":
                break
            print(f"{entry['id']}: {batch.processing_status}, {batch.request_counts.processing} processing")
            time.sleep(60)

        kept = skipped = 0
        with CAPTIONS.open("a", encoding="utf-8") as out:
            for result in client.messages.batches.results(entry["id"]):
                if result.result.type != "succeeded":
                    skipped += 1
                    continue
                msg = result.result.message
                # refusal / max_tokens leave no valid JSON: those items just stay uncaptioned
                if msg.stop_reason != "end_turn":
                    skipped += 1
                    continue
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    items = json.loads(text)["items"]
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
                    continue
                for item in items:
                    if item["id"] in valid_ids and item["en"] and item["fr"]:
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        kept += 1
        entry["fetched"] = True
        STATE.write_text(json.dumps(state, indent=2))
        print(f"{entry['id']}: {kept} kaomojis captioned, {skipped} requests skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--model", default="claude-haiku-4-5")
    s.add_argument("--limit", type=int, default=0, help="caption only the first N (cheap trial run)")
    s.set_defaults(func=submit)
    sub.add_parser("fetch").set_defaults(func=fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
