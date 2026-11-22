"""Дополнительный сбор полных текстов с сайтов, которые отдают статью по HTTP.

Forbes остаётся в корпусе. Сюда добавлены аналогичные источники из условия
и обсуждения: arXiv (q-fin), TechCrunch и CNBC. Запись не затирает уже
собранный текст Forbes, если адрес совпал.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from collect import AI_HINT, FINANCE_HINT, paragraphs, write_corpus

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "publications.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 "
        "RGR-variant4-research"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_QUERIES = [
    'all:"machine learning" AND all:trading AND cat:q-fin.*',
    'all:"artificial intelligence" AND (all:finance OR all:banking OR all:portfolio) AND cat:q-fin.*',
    'all:"deep learning" AND (all:stock OR all:market) AND cat:q-fin.*',
    'ti:"algorithmic trading" AND (all:"machine learning" OR all:"neural")',
]
FEEDS = [
    ("TechCrunch", "https://techcrunch.com/category/fintech/feed/"),
    ("TechCrunch", "https://techcrunch.com/tag/artificial-intelligence/feed/"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147"),
    ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
]


def session() -> requests.Session:
    http = requests.Session()
    http.headers.update(HEADERS)
    return http


def arxiv_rows(http: requests.Session) -> list[dict]:
    rows = []
    seen = set()
    for query in ARXIV_QUERIES:
        response = http.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "start": 0, "max_results": 20, "sortBy": "submittedDate", "sortOrder": "descending"},
            timeout=40,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        batch = root.findall("a:entry", ATOM)
        print(f"arXiv {len(batch):2d}  {query[:70]}")
        for entry in batch:
            title = " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split())
            abstract = " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split())
            url = ""
            for link in entry.findall("a:link", ATOM):
                if link.get("type") == "text/html" or link.get("rel") == "alternate":
                    url = link.get("href") or url
            if not url:
                url = entry.findtext("a:id", default="", namespaces=ATOM)
            published = (entry.findtext("a:published", default="", namespaces=ATOM) or "")[:10]
            key = re.sub(r"\W+", " ", title.lower()).strip()
            blob = f"{title} {abstract}"
            if not title or key in seen or not (AI_HINT.search(blob) and FINANCE_HINT.search(blob)):
                continue
            seen.add(key)
            rows.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                    "query": "arxiv:" + query[:80],
                    "text_source": "arxiv",
                    "text": abstract,
                    "n_chars": len(abstract),
                }
            )
        time.sleep(3)
    return rows[:40]


def feed_rows(http: requests.Session) -> list[dict]:
    rows = []
    seen = set()
    for site, url in FEEDS:
        try:
            response = http.get(url, timeout=30)
            response.raise_for_status()
            xml = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)\w+;", " ", response.text)
            root = ET.fromstring(xml)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"{site} feed error: {exc}")
            continue
        items = root.findall(".//item")
        print(f"{site:12} {len(items):3d}  {url[:70]}")
        kept = 0
        for item in items:
            title = " ".join((item.findtext("title") or "").split())
            link = (item.findtext("link") or "").strip()
            description = " ".join(re.sub(r"<[^>]+>", " ", item.findtext("description") or "").split())
            key = re.sub(r"\W+", " ", title.lower()).strip()
            blob = f"{title} {description}"
            if not title or not link or key in seen or not (AI_HINT.search(blob) and FINANCE_HINT.search(blob)):
                continue
            try:
                page = http.get(link, timeout=25)
            except requests.RequestException:
                continue
            if page.status_code != 200 or len(page.text) < 5000:
                print(f"  skip {page.status_code} {title[:60]}")
                continue
            text = paragraphs(page.text)
            if len(text) < 500 or not (AI_HINT.search(text) and FINANCE_HINT.search(text)):
                continue
            seen.add(key)
            kept += 1
            published = ""
            raw_date = item.findtext("pubDate") or ""
            match = re.search(r"(20\d{2})", raw_date)
            if match:
                from email.utils import parsedate_to_datetime
                try:
                    published = parsedate_to_datetime(raw_date).date().isoformat()
                except (TypeError, ValueError):
                    published = ""
            rows.append(
                {
                    "title": title,
                    "url": link,
                    "published": published,
                    "query": url,
                    "text_source": site.lower().replace(" ", ""),
                    "text": text,
                    "n_chars": len(text),
                }
            )
            if kept >= 15:
                break
        print(f"  kept {kept}")
    return rows


def main() -> None:
    http = session()
    fresh = arxiv_rows(http) + feed_rows(http)
    existing = json.loads(DATA.read_text(encoding="utf-8")).get("articles", []) if DATA.exists() else []
    by_url = {row["url"]: row for row in existing}
    added = 0
    for row in fresh:
        if row["url"] in by_url and by_url[row["url"]].get("n_chars", 0) >= row["n_chars"]:
            continue
        by_url[row["url"]] = row
        added += 1
    records = list(by_url.values())
    write_corpus(records)
    print(
        f"new or replaced {added}; arxiv {sum(1 for r in fresh if r['text_source']=='arxiv')}; "
        f"pages {sum(1 for r in fresh if r['text_source']!='arxiv')}"
    )


if __name__ == "__main__":
    main()
