"""Сбор публикаций Forbes о роли ИИ в финансах и трейдинге.

Вариант 4. Сначала читаются открытые RSS-ленты разделов Forbes.
Страница статьи и поиск сайта отвечают 403. Полный текст, если он не
открылся, добирается по списку Google News с site:forbes.com и по снимку
Wayback Machine. Короткий лид из RSS сохраняется отдельно и в частотный
анализ не входит: для него мало текста.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Разделы, у которых 18.11.2026 открыт RSS. Страница поиска и HTML статьи
# закрыты капчей и отвечают 403.
FEEDS = ["business", "innovation", "asia"]

QUERIES = [
    'site:forbes.com "artificial intelligence" (trading OR "stock market" OR portfolio OR "hedge fund")',
    'site:forbes.com "machine learning" (bank OR banking OR fintech OR lending OR fraud OR credit)',
    'site:forbes.com ("generative AI" OR ChatGPT OR LLM) (finance OR bank OR investment OR insurance OR trading)',
    'site:forbes.com ("algorithmic trading" OR "AI trading" OR "quantitative trading" OR "robo-advisor")',
    'site:forbes.com ("artificial intelligence" OR "machine learning") (compliance OR regulation OR risk) (bank OR financial OR market)',
    'site:forbes.com (fintech OR "wealth management" OR "asset management" OR payments) ("artificial intelligence" OR "machine learning")',
]

AI_HINT = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|deep learning|"
    r"generative|llm|chatgpt|neural|algorithmic|algo|quant)\b",
    re.I,
)
FINANCE_HINT = re.compile(
    r"\b(trad\w+|markets?|bank\w*|financ\w+|invest\w+|portfolio|hedge|fintech|"
    r"credit|fraud|stocks?|assets?|wealth|insurance|payments?|lend\w+|risk|"
    r"funds?|quant|portfolio|wall street|advisor|adviser)\b",
    re.I,
)
EXCLUDE = re.compile(
    r"\b(seo|marketing|airline|airlines|intellectual property|movie|film|"
    r"rotten tomatoes|celebrity|recipe|fashion)\b",
    re.I,
)
BOILER = re.compile(
    r"^(subscribe|sign in|advertisement|more from forbes|editorial standards|"
    r"play\b|follow |recommended |share this)",
    re.I,
)


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def published_from_url(url: str) -> str:
    match = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", url)
    if not match:
        return ""
    return "-".join(match.groups())


def fetch_forbes_feeds() -> list[dict]:
    """Прямой HTTP-запрос к RSS Forbes. В ленте есть заголовок и лид, не полный текст."""
    rows = []
    seen = set()
    for name in FEEDS:
        url = f"https://www.forbes.com/{name}/feed/"
        try:
            response = session().get(url, timeout=25)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"Forbes RSS {name}: {exc}")
            continue
        items = root.findall(".//item")
        print(f"Forbes /{name}/feed/  {len(items)} материалов")
        for item in items:
            title = re.sub(r"\s+", " ", item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            description = re.sub(r"\s+", " ", description).strip()
            key = re.sub(r"\W+", " ", title.lower()).strip()
            if not title or not link or key in seen:
                continue
            blob = f"{title} {description}"
            if not (AI_HINT.search(blob) and FINANCE_HINT.search(blob)):
                continue
            if EXCLUDE.search(title):
                continue
            seen.add(key)
            rows.append(
                {
                    "title": title,
                    "url": link,
                    "published": published_from_url(link),
                    "query": f"forbes.com/{name}/feed/",
                    "description": description,
                }
            )
    return rows


def fetch_rss(query: str) -> list[dict]:
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    response = session().get(url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item"):
        title = re.sub(r"\s+-\s+Forbes.*$", "", item.findtext("title") or "").strip()
        link = item.findtext("link") or ""
        published = item.findtext("pubDate") or ""
        if not title or "/articles/" not in link:
            continue
        rows.append(
            {
                "title": title,
                "google_link": link,
                "published_raw": published,
                "query": query,
            }
        )
    return rows


def decode_forbes_url(google_link: str) -> str | None:
    art_id = google_link.split("/articles/")[1].split("?")[0]
    page = session().get(f"https://news.google.com/rss/articles/{art_id}", timeout=30)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    node = soup.find(attrs={"data-n-a-sg": True})
    if node is None:
        return None
    signature = str(node["data-n-a-sg"]).replace("\\", "\\\\").replace('"', '\\"')
    timestamp = str(node["data-n-a-ts"])
    inner = (
        '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,'
        'null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],'
        f'"{art_id}",{timestamp},"{signature}"]'
    )
    payload = "f.req=" + urllib.parse.quote(json.dumps([[["Fbv4je", inner]]]))
    resp = session().post(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
        timeout=30,
    )
    resp.raise_for_status()
    match = re.search(r"https://www\.forbes\.com[^\"\\]+", resp.text)
    return match.group(0) if match else None


def paragraphs(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside"]):
        tag.decompose()
    chunks = []
    for node in soup.find_all("p"):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
        if len(text) < 80 or BOILER.search(text):
            continue
        chunks.append(text)
    # Соседние дубли на архивной странице встречаются часто.
    unique = []
    seen = set()
    for chunk in chunks:
        key = chunk[:180]
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return "\n".join(unique)


def fetch_text(url: str) -> tuple[str, str]:
    """Возвращает (текст, источник): forbes, archive или пустую строку."""
    direct = session().get(url, timeout=15)
    if direct.status_code == 200 and "forbes.com" in direct.url and len(direct.text) > 5000:
        text = paragraphs(direct.text)
        if len(text) > 400:
            return text, "forbes"
    snap = "https://web.archive.org/web/20261116000000id_/" + url
    archived = session().get(snap, timeout=45)
    if archived.status_code == 200 and len(archived.text) > 8000:
        text = paragraphs(archived.text)
        if len(text) > 400:
            return text, "archive"
    return "", ""


def title_score(title: str) -> int:
    if EXCLUDE.search(title) or re.search(
        r"attention economy|portfolio of|legendary brands|women’s lives|women's lives",
        title,
        re.I,
    ):
        return 0
    if "forbes councils" in title.lower() and not AI_HINT.search(title):
        return 0
    ai = 3 if AI_HINT.search(title) else 0
    finance = 3 if FINANCE_HINT.search(title) else 0
    # Голое слово risk без финансового контекста даёт ложные заголовки.
    if finance and not re.search(
        r"trad\w+|bank\w*|financ\w+|invest\w+|portfolio|hedge|fintech|credit|"
        r"fraud|stocks?|wealth|insurance|payments?|lend\w+|wall street|"
        r"advisor|adviser|quant|bond|market",
        title,
        re.I,
    ):
        finance = 0
    trading = 4 if re.search(r"trad\w+|quant|hedge|portfolio|fintech|fraud|credit", title, re.I) else 0
    if ai == 0 or finance == 0:
        return 0
    return ai + finance + trading


def relevant(title: str, text: str) -> bool:
    if title_score(title) <= 0:
        return False
    if not text or text.strip() == title.strip():
        return True
    blob = f"{title}\n{text[:2500]}"
    return bool(AI_HINT.search(blob) and FINANCE_HINT.search(blob))


def parse_date(raw: str) -> str:
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return ""


def collect_candidates() -> list[dict]:
    seen = set()
    rows = []
    for query in QUERIES:
        try:
            batch = fetch_rss(query)
        except requests.RequestException as exc:
            print(f"RSS error: {exc}")
            continue
        print(f"RSS {len(batch):3d}  {query[:70]}")
        for row in batch:
            key = re.sub(r"\W+", " ", row["title"].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            row["published"] = parse_date(row["published_raw"])
            rows.append(row)
    scored = []
    for row in rows:
        score = title_score(row["title"])
        if score <= 0:
            continue
        row["score"] = score
        scored.append(row)
    # Сначала более предметные заголовки, внутри группы — более свежие.
    scored.sort(key=lambda row: (row["score"], row["published"]), reverse=True)
    return scored


def enrich(row: dict) -> dict | None:
    try:
        url = decode_forbes_url(row["google_link"])
    except requests.RequestException:
        url = None
    if not url:
        return None
    try:
        text, source = fetch_text(url)
    except requests.RequestException:
        text, source = "", ""
    if not relevant(row["title"], text):
        return None
    return {
        "title": row["title"],
        "url": url,
        "published": row["published"],
        "query": row["query"],
        "text_source": source or "title",
        "text": text or row["title"],
        "n_chars": len(text or row["title"]),
    }


def write_corpus(records: list[dict]) -> None:
    records.sort(key=lambda row: row.get("published") or "", reverse=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_candidates": len(records),
        "n_kept": len(records),
        "n_with_body": sum(1 for row in records if row["n_chars"] >= 500),
        "articles": records,
    }
    json_path = DATA / "publications.json"
    csv_path = DATA / "publications.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["published", "title", "url", "text_source", "n_chars", "query", "text"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"saved {len(records)} articles, bodies {payload['n_with_body']} -> {json_path}")


def merge_forbes_feeds(existing: list[dict] | None = None) -> list[dict]:
    """Добавляет свежие материалы из RSS Forbes, не затирая уже собранный полный текст."""
    by_url = {row["url"]: row for row in (existing or [])}
    fresh = fetch_forbes_feeds()
    print(f"Forbes feed, по теме: {len(fresh)}")
    added = 0
    for row in fresh:
        current = by_url.get(row["url"])
        if current and current.get("text_source") in {"forbes", "archive"} and current.get("n_chars", 0) >= 500:
            continue
        try:
            text, source = fetch_text(row["url"])
        except requests.RequestException:
            text, source = "", ""
        body = text or row["description"] or row["title"]
        by_url[row["url"]] = {
            "title": row["title"],
            "url": row["url"],
            "published": row["published"],
            "query": row["query"],
            "text_source": source or "rss",
            "text": body,
            "n_chars": len(body),
        }
        added += 1
        print(f"  {source or 'rss':8} {len(body):5}  {row['title'][:70]}")
    print(f"обновлено или добавлено из ленты Forbes: {added}")
    return list(by_url.values())


def main() -> None:
    started = time.time()
    json_path = DATA / "publications.json"
    existing = []
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8")).get("articles", [])
    records = merge_forbes_feeds(existing)
    if "--google" in sys.argv:
        by_url = {row["url"]: row for row in records}
        candidates = collect_candidates()[:100]
        print(f"titles after filter: {len(candidates)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(enrich, row) for row in candidates]
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                if not record:
                    continue
                current = by_url.get(record["url"])
                if current and current.get("n_chars", 0) >= record["n_chars"]:
                    continue
                by_url[record["url"]] = record
                if done % 10 == 0:
                    print(f"  processed {done}/{len(candidates)}")
        records = list(by_url.values())
    write_corpus(records)
    print(f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
