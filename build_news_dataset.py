import csv, re, feedparser
from datetime import datetime, timedelta, timezone

OUT_CSV = "news_300_dataset.csv"
TARGET_PER_CAT = {"Business": 100, "Health": 100, "Politics": 100}
CUTOFF = datetime.now(timezone.utc) - timedelta(days=730)  # last 2 years

FEEDS = {
    "Business": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.ft.com/?format=rss",
        "https://www.economist.com/business/rss.xml",
        "https://www.forbes.com/business/feed/",
        "https://www.inc.com/rss.xml",
        "https://www.wsj.com/xml/rss/3_7014.xml",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.businessinsider.com/rss"
    ],
    "Health": [
        "https://feeds.bbci.co.uk/news/health/rss.xml",
        "https://www.medicalnewstoday.com/rss",
        "https://www.nih.gov/news-events/news-releases.xml",
        "https://www.sciencedaily.com/rss/health_medicine.xml",
        "https://www.health.harvard.edu/blog/feed",
        "https://www.cdc.gov/media/rss.htm",
        "https://www.webmd.com/rss/news_breaking.xml",
        "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml"
    ],
    "Politics": [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://www.politico.com/rss/politics08.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "https://www.npr.org/rss/rss.php?id=1014",
        "https://www.reuters.com/politics/rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.cnn.com/rss/cnn_allpolitics.rss",
        "https://www.theguardian.com/politics/rss"
    ]
}

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '')

def clean_text(t):
    t = strip_html(t)
    return re.sub(r"\s+", " ", t).strip() if t else ""

def to_filename(title, source, date):
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{base[:80]}_{date.strftime('%Y%m%d')}.txt"

def parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def fetch_feed(url, category):
    rows = []
    feed = feedparser.parse(url)
    source = feed.feed.get("title", "Unknown Source")
    for e in feed.entries:
        dt = parse_date(e)
        if dt < CUTOFF:
            continue
        title = clean_text(e.get("title"))
        link = e.get("link")
        author = clean_text(e.get("author", "")) or source
        summary = ""
        if "content" in e and isinstance(e["content"], list) and e["content"]:
            summary = " ".join(clean_text(part.get("value", "")) for part in e["content"])
        if not summary:
            summary = clean_text(e.get("summary", "")) or clean_text(e.get("description", ""))
        summary = summary[:1000]
        if title and link:
            rows.append({
                "filename": to_filename(title, source, dt),
                "category": category,
                "title": title,
                "author": author,
                "date": dt.strftime("%Y-%m-%d"),
                "source": source,
                "url": link,
                "summary": summary
            })
    return rows

def build_dataset():
    all_rows = []
    seen_links = set()
    for cat, urls in FEEDS.items():
        cat_rows = []
        for u in urls:
            try:
                cat_rows.extend(fetch_feed(u, cat))
            except Exception as ex:
                print(f"Failed {u}: {ex}")
        # Remove duplicates by URL
        unique_cat_rows = []
        seen_titles = set()
        for row in cat_rows:
            if row["url"] not in seen_links and row["title"].lower() not in seen_titles:
                unique_cat_rows.append(row)
                seen_links.add(row["url"])
                seen_titles.add(row["title"].lower())
        unique_cat_rows.sort(key=lambda x: x["date"], reverse=True)
        all_rows.extend(unique_cat_rows[:TARGET_PER_CAT[cat]])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","category","title","author","date","source","url","summary"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUT_CSV}")
    counts = {c: sum(1 for r in all_rows if r["category"] == c) for c in TARGET_PER_CAT}
    print("Counts:", counts)

if __name__ == "__main__":
    build_dataset()
