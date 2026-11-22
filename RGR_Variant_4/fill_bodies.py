"""Добирает полный текст статей, для которых прямой запрос к Forbes вернул отказ.

Берётся ближайший успешный снимок Wayback Machine по точному timestamp из CDX.
Снимок принимается, только если в тексте есть содержательные слова заголовка:
так отсекаются чужие страницы по тому же адресу.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
from pathlib import Path

import requests

from collect import paragraphs

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "data" / "publications.json"
CSV_PATH = ROOT / "data" / "publications.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
LOCK = threading.Lock()
LAST = {"t": 0.0}


def pace() -> None:
    with LOCK:
        wait = 1.1 - (time.time() - LAST["t"])
        if wait > 0:
            time.sleep(wait)
        LAST["t"] = time.time()


def get(url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        pace()
        response = requests.get(url, headers=HEADERS, timeout=50, **kwargs)
        if response.status_code != 429:
            return response
        time.sleep(8 + attempt * 4)
    return response


def latest_timestamp(url: str) -> str:
    response = get(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": url,
            "output": "json",
            "fl": "timestamp",
            "filter": "statuscode:200",
            "sort": "reverse",
            "limit": "1",
        },
    )
    response.raise_for_status()
    rows = response.json()
    if len(rows) < 2:
        return ""
    return rows[1][0]


def title_words(title: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{5,}", title.lower())
    stop = {"about", "after", "their", "there", "which", "would", "could", "these", "those", "using"}
    return [word for word in words if word not in stop][:8]


def matches_title(title: str, text: str) -> bool:
    words = title_words(title)
    if not words:
        return False
    lowered = text.lower()
    hits = sum(1 for word in words if word in lowered)
    return hits >= min(2, len(words))


def fill_one(article: dict) -> dict:
    if article["text_source"] != "title":
        return article
    try:
        stamp = latest_timestamp(article["url"])
        if not stamp:
            return article
        snap = f"https://web.archive.org/web/{stamp}id_/{article['url']}"
        page = get(snap)
        if page.status_code != 200 or len(page.text) < 8000:
            return article
        text = paragraphs(page.text)
        if len(text) < 500 or not matches_title(article["title"], text):
            return article
        article["text"] = text
        article["text_source"] = "archive"
        article["n_chars"] = len(text)
        return article
    except (requests.RequestException, ValueError, json.JSONDecodeError):
        return article


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    articles = payload["articles"]
    pending = [article for article in articles if article["text_source"] == "title"]
    print(f"to fill: {len(pending)}")
    filled = 0
    for index, article in enumerate(pending, start=1):
        before = article["text_source"]
        fill_one(article)
        if article["text_source"] != before:
            filled += 1
            print(f"  ok {filled:02d} {article['n_chars']:5d}  {article['title'][:70]}")
        elif index % 10 == 0:
            print(f"  ... {index}/{len(pending)}, filled {filled}")
    payload["n_with_body"] = sum(1 for article in articles if article["text_source"] != "title")
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["published", "title", "url", "text_source", "n_chars", "query", "text"],
        )
        writer.writeheader()
        writer.writerows(articles)
    print(f"bodies now {payload['n_with_body']}")


if __name__ == "__main__":
    main()
