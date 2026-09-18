"""kaomoji.ru: one page, categories are <h3> headings followed by kaomoji tables."""
from bs4 import BeautifulSoup

from .common import RAW_DIR, Fetcher, write_jsonl

URL = "https://kaomoji.ru/en/"


def main() -> None:
    html = Fetcher().get(URL)
    if html is None:
        raise SystemExit("could not fetch kaomoji.ru")
    soup = BeautifulSoup(html, "lxml")

    records, group, category = [], None, None
    for el in soup.find_all(["h2", "h3", "table"]):
        if el.name == "h2":
            group = el.get_text(" ", strip=True).removeprefix("Japanese Emoticons:").strip()
        elif el.name == "h3":
            category = el.get_text(" ", strip=True)
        elif "table_kaomoji" in (el.get("class") or []) and category:
            # The "Special" table is (kaomoji, human-written caption) pairs; others are all kaomoji.
            if category == "Special":
                cells = [(tds[0], tds[1].get_text(" ", strip=True))
                         for tr in el.find_all("tr") if len(tds := tr.find_all("td")) == 2]
            else:
                cells = [(td, None) for td in el.find_all("td")]
            for td, caption in cells:
                text = td.get_text(strip=True)
                if text:
                    records.append({
                        "text": text,
                        "tags": [category.lower()],
                        "category": f"{group} / {category}" if group else category,
                        "caption": caption,
                        "nsfw": False,
                        "source": "kaomoji.ru",
                    })
    write_jsonl(RAW_DIR / "kaomoji_ru.jsonl", records)


if __name__ == "__main__":
    main()
