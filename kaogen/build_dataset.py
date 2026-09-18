"""Merge raw scrapes into data/kaomojis.jsonl: normalize, classify, filter, dedupe."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from wcwidth import wcswidth

RAW_DIR = Path("data/raw")
OUT = Path("data/kaomojis.jsonl")

EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U0001F1E6-\U0001F1FF‍️]")
BRAILLE_RE = re.compile("[⠀-⣿]")


# emojicombos doubles as a social-media bio/copypasta board. Those entries are text, not drawings:
# training on them teaches the model to emit fill-in-the-blank profile templates.
WATERMARK_RE = re.compile(
    r"(search\s+\S+\s+for\s+more|for\s+more\s+search|for more\s*[!！]|credits?\s+(are|to)\b)[^\n]*",
    re.I)
PLACEHOLDER_RE = re.compile(
    r"\b(prns?|mbti|@user|ur name|your name|status|info|bio here|url|dm me|followers|"
    r"read please|join my|link in|my group|add me|vanity|profile|text|txt|name here|"
    r"insert name|your text|here)\b"
    # a parenthesised slot to fill in: "(name)", "(age)", "(pronouns)". Bare "name" is not
    # usable as a signal -- it appears in real drawings -- but the bracketed form always is.
    r"|\(\s*(name|age|pronouns?|bff|nickname|user|username)\s*\)", re.I)
LATIN_RE = re.compile(r"[a-zA-Z]")


def strip_watermark(text: str) -> str:
    """Remove a "search X for more" / credits promo from the end of the line carrying it.

    Uploaders stamp real drawings with a promo phrase, often on the same line as the art itself,
    so neither dropping the entry nor dropping the line is safe: cut the phrase only.
    """
    return WATERMARK_RE.sub("", text)


# Hangul filler, braille blank, zero-width and format characters: invisible, but str.isspace()
# says False, so an entry made only of these looks non-empty and reaches the training set.
INVISIBLE = {"\u3164", "\u2800", "\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0", "\u115f", "\u1160"}


def is_blank(text: str) -> bool:
    return all(c.isspace() or c in INVISIBLE or unicodedata.category(c) == "Cf" for c in text)


def is_bio_template(text: str) -> bool:
    """True for fill-in-the-blank profile templates.

    Their placeholders are often written in styled Unicode letters (𝗇𝖺𝗆𝖾, 𝑡𝑒𝑥𝑡) that a plain
    [a-z] pattern never sees, so match against an NFKC fold. The fold is for detection only:
    NFKC would flatten the fullwidth forms kaomojis are built from, so it never reaches storage.
    """
    return PLACEHOLDER_RE.search(unicodedata.normalize("NFKC", text)) is not None


def is_prose(text: str) -> bool:
    """True when the entry is mostly words rather than a drawing.

    Kaomojis do carry short latin captions ("I love you", "nope"), so the cutoff is generous and
    only applies to long entries; short ones can never trip it.
    """
    dense = [c for c in text if not c.isspace()]
    if len(dense) <= 25:
        return False
    if len(LATIN_RE.findall(text)) > 0.45 * len(dense):
        return True
    # A chat message framed by box-drawing rules dilutes the whole-entry ratio below the cutoff.
    # Count lines that are sentences in their own right instead -- but a drawing is allowed one
    # caption line ("Turn up the music!"), so only a majority of such lines means prose.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    sentences = 0
    for line in lines:
        packed = [c for c in line if not c.isspace()]
        words = re.findall(r"[a-zA-Z]{2,}", line)
        if len(words) >= 3 and len(LATIN_RE.findall(line)) > 0.7 * len(packed):
            sentences += 1
    return sentences >= 2 and sentences >= 0.4 * len(lines)


def normalize(text: str) -> str:
    # NFC only: NFKC would fold the fullwidth/halfwidth forms kaomojis are built from.
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    # Dedent by the longest whitespace prefix shared by all non-blank lines (keeps alignment).
    body = [ln for ln in lines if ln.strip()]
    prefix = body[0][: len(body[0]) - len(body[0].lstrip())]
    for ln in body[1:]:
        while prefix and not ln.startswith(prefix):
            prefix = prefix[:-1]
    return "\n".join(ln[len(prefix):] if ln.startswith(prefix) else ln for ln in lines)


def width(text: str) -> int:
    return max((w if (w := wcswidth(ln)) >= 0 else len(ln)) for ln in text.split("\n"))


def is_zalgo(text: str) -> bool:
    # Stacked combining marks are zero-width, so they slip past the width limit.
    # ( ͡° ͜ʖ ͡°) is ~27% marks; zalgo text is well above half.
    marks = sum(1 for c in text if unicodedata.category(c) in ("Mn", "Me"))
    return marks > 0.5 * sum(1 for c in text if not c.isspace())


def style(text: str) -> str:
    chars = [c for c in text if not c.isspace()]
    if len(EMOJI_RE.findall(text)) > 0.5 * len(chars):
        return "emoji"
    if len(BRAILLE_RE.findall(text)) > 0.5 * len(chars):
        return "braille"
    return "multiline" if "\n" in text else "kaomoji"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=24)
    ap.add_argument("--max-width", type=int, default=64)
    ap.add_argument("--keep-styles", default="kaomoji,multiline,braille")
    args = ap.parse_args()
    keep = set(args.keep_styles.split(","))

    merged: dict[str, dict] = {}
    dropped = Counter()
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            text = normalize(strip_watermark(rec["text"]))
            if not text or is_blank(text):
                dropped["empty"] += 1
                continue
            st = style(text)
            n_lines, w = text.count("\n") + 1, width(text)
            if st not in keep:
                dropped[f"style:{st}"] += 1
                continue
            if n_lines > args.max_lines or w > args.max_width or len(text) > 2 * n_lines * args.max_width:
                dropped["too_big"] += 1
                continue
            if is_zalgo(text):
                dropped["zalgo"] += 1
                continue
            if is_bio_template(text):
                dropped["bio template"] += 1
                continue
            if is_prose(text):
                dropped["prose, not a drawing"] += 1
                continue
            if n_lines == 1 and len(text) < 2:
                dropped["too_short"] += 1
                continue
            cur = merged.get(text)
            if cur is None:
                merged[text] = {
                    "id": hashlib.sha1(text.encode()).hexdigest()[:12],
                    "text": text,
                    "tags": list(dict.fromkeys(rec["tags"])),
                    "captions": [rec["caption"]] if rec.get("caption") else [],
                    "nsfw": rec["nsfw"],
                    "style": st,
                    "n_lines": n_lines,
                    "width": w,
                    "sources": [rec["source"]],
                }
            else:
                dropped["duplicate"] += 1
                cur["tags"] = list(dict.fromkeys(cur["tags"] + rec["tags"]))
                cur["nsfw"] = cur["nsfw"] or rec["nsfw"]
                if rec.get("caption") and rec["caption"] not in cur["captions"]:
                    cur["captions"].append(rec["caption"])
                if rec["source"] not in cur["sources"]:
                    cur["sources"].append(rec["source"])

    records = list(merged.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"kept {len(records)} -> {OUT}")
    print("dropped:", dict(dropped))
    print("by style:", dict(Counter(r["style"] for r in records)))
    print("nsfw:", sum(r["nsfw"] for r in records))
    print("by source:", dict(Counter(s for r in records for s in r["sources"])))
    ml = sorted(r["n_lines"] for r in records if r["n_lines"] > 1)
    if ml:
        print(f"multi-line height: median {ml[len(ml)//2]}, p90 {ml[int(len(ml)*0.9)]}, max {ml[-1]}")


if __name__ == "__main__":
    main()
