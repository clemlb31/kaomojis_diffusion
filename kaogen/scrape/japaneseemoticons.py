"""japaneseemoticons.me: one page per category, <h3> subcategories, paginated."""
import argparse
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .common import RAW_DIR, Fetcher, write_jsonl

HOME = "https://japaneseemoticons.me/"
CATEGORY_RE = re.compile(r"^https://japaneseemoticons\.me/[a-z0-9-]*emoticon[a-z0-9-]*/$")


def parse_page(soup: BeautifulSoup, category: str) -> list[dict]:
    records, sub = [], None
    content = soup.find("article") or soup
    for el in content.find_all(["h3", "td"]):
        if el.name == "h3":
            sub = el.get_text(" ", strip=True)
            continue
        text = el.get_text(strip=True)
        if text:
            tags = [category.lower()] + ([sub.lower()] if sub else [])
            records.append({
                "text": text,
                "tags": tags,
                "category": f"{category} / {sub}" if sub else category,
                "nsfw": False,
                "source": "japaneseemoticons.me",
            })
    return records


MAX_CONSECUTIVE_FAILURES = 3  # each already retried with backoff: the site stalls intermittently


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=5.0)
    args = ap.parse_args()
    fetcher = Fetcher(delay=args.delay, retries=3, backoff=30, timeout=15)
    failures = 0
    home = fetcher.get(HOME)
    if home is None:
        raise SystemExit("could not fetch japaneseemoticons.me")
    soup = BeautifulSoup(home, "lxml")
    categories = sorted({a["href"] for a in soup.find_all("a", href=True) if CATEGORY_RE.match(a["href"])})
    print(f"{len(categories)} categories")

    records = []
    for cat_url in categories:
        todo, seen = [cat_url], set()
        while todo:
            url = todo.pop(0)
            if url in seen:
                continue
            seen.add(url)
            html = fetcher.get(url)
            if html is None:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"{failures} consecutive failures, stopping early (rerun later: cached pages are kept)")
                    write_jsonl(RAW_DIR / "japaneseemoticons.jsonl", records)
                    return
                continue
            failures = 0
            page = BeautifulSoup(html, "lxml")
            h1 = page.find("h1")
            name = (h1.get_text(" ", strip=True) if h1 else cat_url.rstrip("/").rsplit("/", 1)[-1])
            name = re.sub(r"\s*(emoticons?|kaomojis?)\s*", " ", name, flags=re.I).strip() or name
            found = parse_page(page, name)
            records.extend(found)
            print(f"  {url}: {len(found)}", flush=True)
            for a in page.select("a.post-page-numbers[href]"):
                nxt = urljoin(url, a["href"])
                if nxt.startswith(cat_url):
                    todo.append(nxt)
    write_jsonl(RAW_DIR / "japaneseemoticons.jsonl", records)


if __name__ == "__main__":
    main()
