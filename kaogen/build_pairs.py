"""Turn data/kaomojis.jsonl (+ optional data/captions.jsonl) into prompt -> kaomoji pairs.

Prompt sources, best first: human captions, LLM captions (EN + FR), then prompts templated
from community tags. Tag prompts are the only source for NSFW items, which are never sent
to the captioning API. The split is by kaomoji, so no drawing leaks from train into val.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATASET = Path("data/kaomojis.jsonl")
CAPTIONS = Path("data/captions.jsonl")
TAGS_FR = json.loads((Path(__file__).parent / "tags_fr.json").read_text(encoding="utf-8"))

# Describe the medium, not the subject: usable as a style hint, never as the only content.
STYLE_TAGS = {"text art", "dot art", "ascii art", "kaomoji", "aesthetic", "emoji", "emoticon", "symbol",
              "symbols", "art", "cute symbols", "cute kaomoji", "face kaomoji", "text", "copy paste",
              "bio", "discord", "new", "unicode", "ascii", "braille", "text face"}

SIZE = {
    "kaomoji": {"en": ["small", "one-line", "tiny"], "fr": ["petit", "sur une ligne", "minuscule"]},
    "multiline": {"en": ["big", "multi-line", "large"], "fr": ["grand", "sur plusieurs lignes", "gros"]},
    "braille": {"en": ["braille dot art", "detailed dot art", "realistic dot art"],
                "fr": ["dot art en braille", "dot art détaillé", "dot art réaliste"]},
}
TEMPLATES = {
    "en": ["{tags}", "{tags} kaomoji", "{tags}, {size}", "draw me a {size} kaomoji: {tags}",
           "make a {tags} text art", "{size} kaomoji, {tags}"],
    # tags are inserted as-is, so French templates avoid anything that needs gender agreement
    "fr": ["{tags}", "kaomoji {tags}", "{tags}, {size}", "fais-moi un kaomoji {size} : {tags}",
           "dessine-moi en ascii : {tags}", "kaomoji {size}, {tags}"],
}


def tag_prompts(rec: dict, rng: random.Random, n: int) -> list[str]:
    content = [t for t in rec["tags"] if t not in STYLE_TAGS and len(t) <= 30 and t.strip()]
    if not content:
        return []
    prompts = []
    for i in range(n):
        lang = "fr" if i % 2 else "en"
        # earlier tags are the ones the uploader thought of first: favour them
        picked = rng.sample(content[:6], k=min(len(content[:6]), rng.randint(1, 3)))
        if lang == "fr":
            picked = [TAGS_FR.get(t, t) for t in picked]
        prompt = rng.choice(TEMPLATES[lang]).format(tags=" ".join(picked), size=rng.choice(SIZE[rec["style"]][lang]))
        prompts.append(prompt)
    return list(dict.fromkeys(prompts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default="kaomoji,multiline", help="comma list among kaomoji,multiline,braille")
    ap.add_argument("--max-lines", type=int, default=16)
    ap.add_argument("--max-width", type=int, default=48)
    ap.add_argument("--tag-prompts", type=int, default=4, help="templated prompts per kaomoji")
    ap.add_argument("--val-percent", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    styles, rng = set(args.styles.split(",")), random.Random(args.seed)

    captions = {}
    if CAPTIONS.exists():
        for line in CAPTIONS.open(encoding="utf-8"):
            c = json.loads(line)
            captions[c["id"]] = c["en"] + c["fr"]

    splits, sources, no_prompt = {"train": [], "val": []}, Counter(), 0
    for line in DATASET.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["style"] not in styles or rec["n_lines"] > args.max_lines or rec["width"] > args.max_width:
            continue
        llm = captions.get(rec["id"], [])
        # with real captions, tag prompts are just augmentation: halve them
        templated = tag_prompts(rec, rng, args.tag_prompts if not llm else args.tag_prompts // 2)
        prompts = list(dict.fromkeys(rec["captions"] + llm + templated))
        if not prompts:
            no_prompt += 1
            continue
        sources.update(human=len(rec["captions"]), llm=len(llm), tags=len(templated))
        split = "val" if int(rec["id"], 16) % 100 < args.val_percent else "train"
        for p in prompts:
            splits[split].append({"prompt": p.strip().lower(), "text": rec["text"], "id": rec["id"],
                                  "style": rec["style"], "nsfw": rec["nsfw"]})

    for name, rows in splits.items():
        rng.shuffle(rows)
        path = Path(f"data/{name}.jsonl")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} pairs, {len({r['id'] for r in rows})} kaomojis, "
              f"{sum(r['nsfw'] for r in rows)} nsfw pairs -> {path}")
    print("prompt sources:", dict(sources), "| skipped (no usable tag or caption):", no_prompt)


if __name__ == "__main__":
    main()
