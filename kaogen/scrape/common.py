"""Polite, disk-cached HTTP fetching shared by all scrapers."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path("data/raw/html")
RAW_DIR = Path("data/raw")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


class Fetcher:
    """GET pages with a minimum delay between network hits; cached pages are free."""

    def __init__(self, delay: float = 2.0, retries: int = 0, backoff: float = 30.0, timeout: float = 30.0):
        self.delay, self.retries, self.backoff, self.timeout = delay, retries, backoff, timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_hit = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_path(url: str) -> Path:
        return CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")

    def is_cached(self, url: str) -> bool:
        return self._cache_path(url).exists()

    def get(self, url: str) -> str | None:
        path = self._cache_path(url)
        if path.exists():
            return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")

        for attempt in range(self.retries + 1):
            if attempt:
                # some sites stall for a minute or two, then recover on their own: wait it out
                time.sleep(self.backoff * attempt)
            text = self._fetch(url)
            if text is not None:
                path.write_bytes(gzip.compress(text.encode("utf-8")))
                return text
        return None

    def _fetch(self, url: str) -> str | None:
        wait = self.delay - (time.monotonic() - self._last_hit)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as e:
            print(f"  ! {url}: {type(e).__name__}", flush=True)
            return None
        finally:
            self._last_hit = time.monotonic()
        if r.status_code != 200:
            print(f"  ! {url}: HTTP {r.status_code}", flush=True)
            return None
        r.encoding = "utf-8"
        return r.text


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {path}")
