"""emojicombos.com: one page per tag; crawl breadth-first by following tag links.

Each combo carries its exact text base64-encoded (whitespace preserved), its tags,
and a "sensitive" marker that we keep as the nsfw flag.
"""
import argparse
import base64
import re
from collections import Counter, deque

from bs4 import BeautifulSoup

from .common import RAW_DIR, Fetcher, write_jsonl

BASE = "https://emojicombos.com"
SLUG_RE = re.compile(r"^/[a-z0-9][a-z0-9-]{0,40}$")

SEEDS = [
    # emotions / actions
    "happy", "sad", "angry", "love", "cry", "shy", "scared", "surprised", "sleepy", "confused",
    "hug", "kiss", "dance", "wave", "table-flip", "shrug", "fight", "running", "hiding", "wink",
    # creatures / things
    "cat", "dog", "bear", "bunny", "bird", "fish", "frog", "dragon", "butterfly", "flower",
    "heart", "star", "moon", "sword", "gun", "music", "food", "coffee", "skull", "ghost",
    # text art
    "ascii-art", "text-art", "dot-art", "kaomoji", "anime", "meme", "aesthetic", "cute",
    # nsfw
    "nsfw", "sexy", "lewd", "horny", "boobs", "ass", "dick", "sex",
]

# Never collected, whatever the context.
BLOCKED_TAGS = {"loli", "lolicon", "shota", "shotacon", "cp", "pedo", "pedophile", "cub"}
# Dropped when the combo is also nsfw.
MINOR_TAGS = {"child", "children", "kid", "kids", "minor", "underage", "teen", "teens",
              "schoolgirl", "schoolboy", "baby", "toddler", "young", "preteen"}
SEXUAL_TAGS = {"nsfw", "sex", "sexy", "lewd", "horny", "porn", "hentai", "boobs", "tits", "ass",
               "dick", "penis", "cock", "pussy", "vagina", "nude", "naked", "cum", "blowjob", "bdsm"}


def _b64(value: str | None) -> str:
    try:
        return base64.b64decode(value or "").decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def parse_keywords(item) -> list[tuple[str, str]]:
    """(tag, href) pairs. Sensitive items hide both behind base64 data attributes until "show" is
    clicked: reading only visible anchors silently yields no tags for exactly the nsfw items."""
    pairs = []
    for a in item.select(".keywords a:not(.new-tag)"):  # .new-tag is a "recently added" badge
        tag = a.get_text(strip=True) or _b64(a.get("data-encoded-text"))
        href = a.get("href") or _b64(a.get("data-encoded-href"))
        if tag.strip():
            pairs.append((tag.strip().lower(), href))
    return pairs


def parse_page(html: str, filtered: Counter | None = None) -> tuple[list[dict], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    records, links = [], []
    for item in soup.select(".combo-ctn[data-encoded-combo]"):
        text = _b64(item["data-encoded-combo"])
        if not text:
            continue
        keywords = parse_keywords(item)
        tags = [tag for tag, _ in keywords]
        tagset = set(tags)
        nsfw = item.select_one(".sensitive-item-blur") is not None or bool(tagset & SEXUAL_TAGS)
        if tagset & BLOCKED_TAGS or (nsfw and tagset & MINOR_TAGS):
            if filtered is not None:
                filtered[item.get("data-combo-hash")] += 1
            continue
        links.extend(href for _, href in keywords if SLUG_RE.match(href))
        records.append({
            "id": item.get("data-combo-hash"),
            "text": text,
            "tags": tags,
            "category": None,
            "nsfw": nsfw,
            "source": "emojicombos.com",
        })
    return records, links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fetches", type=int, default=400,
                    help="network budget; cached pages are free, so 0 re-parses the cache offline")
    ap.add_argument("--seeds", default="", help="comma list of extra tags to crawl first")
    ap.add_argument("--delay", type=float, default=2.5)
    args = ap.parse_args()

    fetcher = Fetcher(delay=args.delay)
    extra = [s.strip().replace(" ", "-") for s in args.seeds.split(",") if s.strip()]
    queue = deque(f"/{s}" for s in extra + SEEDS)
    seen_pages, by_id, fetches, filtered = set(), {}, 0, Counter()
    while queue:
        slug = queue.popleft()
        if slug in seen_pages or slug.strip("/") in BLOCKED_TAGS:
            continue
        seen_pages.add(slug)
        if not fetcher.is_cached(BASE + slug):
            if fetches >= args.max_fetches:
                continue
            fetches += 1
        html = fetcher.get(BASE + slug)
        if html is None:
            continue
        records, links = parse_page(html, filtered)
        new = 0
        for rec in records:
            if rec["id"] not in by_id:
                by_id[rec["id"]] = rec
                new += 1
        queue.extend(l for l in links if l not in seen_pages)
        print(f"[{fetches}/{args.max_fetches} fetched] {slug}: {len(records)} combos, {new} new (total {len(by_id)})", flush=True)

    print(f"content filter dropped {len(filtered)} distinct combos (blocked tags, or minor-related tags on nsfw items)")
    write_jsonl(RAW_DIR / "emojicombos.jsonl", list(by_id.values()))


if __name__ == "__main__":
    main()
