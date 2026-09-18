"""Caption kaomojis by hand (or by the assistant reading this repo), no API key needed.

    python -m kaogen.caption_manual next --n 40      # print the next batch to describe
    python -m kaogen.caption_manual add < lines.txt  # ingest compact caption lines

The compact line format avoids JSON escaping, which is error-prone to write by hand:

    <id>|<en1>;<en2>;<en3>|<fr1>;<fr2>;<fr3>

Items already present in data/captions.jsonl are never handed out again, so `next` can be
called repeatedly to walk the queue. Priority: multi-line art first (its tags describe the
subject but never the scene), then one-liners whose tag list is too thin for templating.
"""
import argparse
import json
import sys
from pathlib import Path

from .build_pairs import STYLE_TAGS

DATASET = Path("data/kaomojis.jsonl")
CAPTIONS = Path("data/captions.jsonl")


def content_tags(rec: dict) -> list[str]:
    return [t for t in rec["tags"] if t not in STYLE_TAGS and len(t) <= 30 and t.strip()]


def priority(rec: dict) -> tuple:
    """Lower sorts first: the worse the tags describe the art, the sooner it needs a caption.

    Within that, art closest to 7 lines first. Two-line pieces are usually just an emoji pair
    (little for a caption to add), 16-line murals cost ten times as much to read for one caption,
    and the drawings that actually depict a scene sit in between.
    """
    n = len(content_tags(rec))
    band = abs(rec["n_lines"] - 7) if rec["style"] == "multiline" else 0
    return (0 if rec["style"] == "multiline" else 1, min(n, 6), band, rec["width"])


def queue() -> list[dict]:
    done = set()
    if CAPTIONS.exists():
        done = {json.loads(l)["id"] for l in CAPTIONS.open(encoding="utf-8")}
    todo = []
    for line in DATASET.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec["id"] in done:
            continue
        if rec["style"] not in ("kaomoji", "multiline") or rec["n_lines"] > 16 or rec["width"] > 48:
            continue  # same eligibility window as build_pairs' defaults
        if rec["style"] == "multiline" or len(content_tags(rec)) <= 2:
            todo.append(rec)
    return sorted(todo, key=priority)


def cmd_next(args) -> None:
    todo = queue()
    batch = todo[: args.n]
    for rec in batch:
        tags = ", ".join(content_tags(rec)[:8]) or "(aucun)"
        print(f"=== {rec['id']} [{rec['style']}, {rec['n_lines']}L x {rec['width']}c]"
              f"{' [NSFW]' if rec['nsfw'] else ''} tags: {tags}")
        print(rec["text"])
    print(f"=== fin du lot : {len(batch)} items, {len(todo) - len(batch)} restants dans la file")


def cmd_add(args) -> None:
    valid = {json.loads(l)["id"] for l in DATASET.open(encoding="utf-8")}
    done = set()
    if CAPTIONS.exists():
        done = {json.loads(l)["id"] for l in CAPTIONS.open(encoding="utf-8")}

    added, skipped = 0, []
    with CAPTIONS.open("a", encoding="utf-8") as out:
        for lineno, line in enumerate(sys.stdin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                skipped.append(f"L{lineno}: {len(parts)} champs au lieu de 3")
                continue
            kid, en, fr = parts[0].strip(), parts[1], parts[2]
            en = [s.strip().lower() for s in en.split(";") if s.strip()]
            fr = [s.strip().lower() for s in fr.split(";") if s.strip()]
            if kid not in valid:
                skipped.append(f"L{lineno}: id inconnu {kid}")
            elif kid in done:
                skipped.append(f"L{lineno}: doublon {kid}")
            elif not en or not fr:
                skipped.append(f"L{lineno}: {kid} manque en ou fr")
            else:
                out.write(json.dumps({"id": kid, "en": en, "fr": fr}, ensure_ascii=False) + "\n")
                done.add(kid)
                added += 1
    print(f"ajouté {added} légendes -> {CAPTIONS}")
    for s in skipped[:20]:
        print("  ! " + s)
    if skipped:
        print(f"  ({len(skipped)} lignes rejetées)")
    print(f"total légendé : {len(done)} | reste dans la file : {len(queue())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next")
    n.add_argument("--n", type=int, default=40)
    n.set_defaults(func=cmd_next)
    sub.add_parser("add").set_defaults(func=cmd_add)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
