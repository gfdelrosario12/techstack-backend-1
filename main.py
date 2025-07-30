import re
import os
import datetime
import httpx
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI

# === Load Env ===
load_dotenv()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not BRAVE_API_KEY or not OPENAI_KEY:
    raise RuntimeError("❌ Missing API keys in .env")

client_ai = AsyncOpenAI(api_key=OPENAI_KEY)

# === Config ===
MAX_CALLS_PER_DAY = 10
MAX_RESULTS_PER_CALL = 10
SUMMARY_BATCH_SIZE = 5

daily_counters = {"date": datetime.date.today(), "calls": 0, "ai_calls": 0}


def reset_counters_if_new_day():
    today = datetime.date.today()
    if daily_counters["date"] != today:
        daily_counters.update({"date": today, "calls": 0, "ai_calls": 0})


def log(msg: str):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def clean_text(text: str) -> str:
    """Remove bullets, emojis, and special characters for clean frontend output."""
    text = re.sub(r"^[\-\•\●\▪\♦\▶\★\*]+\s*", "", text)
    text = re.sub(r"[^\w\s\.,:;!?/()-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# === Brave Search ===
async def brave_search(query: str) -> List[Dict]:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    async with httpx.AsyncClient(timeout=15) as client:
        log(f"🔍 Searching Brave: {query}")
        res = await client.get(url, headers=headers, params={"q": query, "count": MAX_RESULTS_PER_CALL})
        res.raise_for_status()
        data = res.json()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "source": "brave",
        }
        for item in data.get("web", {}).get("results", [])[:MAX_RESULTS_PER_CALL]
    ]


# === AI Rating ===
async def ai_rate_result(title: str, snippet: str, domain: str, types: List[str]) -> float:
    reset_counters_if_new_day()
    if daily_counters["ai_calls"] >= MAX_CALLS_PER_DAY:
        return 0.5
    content = f"Domain: {domain}\nTypes: {', '.join(types)}\nContent: {title} {snippet[:400]}"
    try:
        resp = await client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rate relevance from 0.0 to 1.0 for this domain and opportunity type."},
                {"role": "user", "content": content},
            ],
            max_tokens=5,
        )
        daily_counters["ai_calls"] += 1
        return float(resp.choices[0].message.content.strip())
    except:
        return 0.5


# === AI Summarizer (Fine-Tuned) ===
async def ai_summarize_headlines(headlines: List[Dict], domain: str, types: List[str]) -> List[Dict]:
    summarized = []
    type_str = ", ".join(types) if types else "tech opportunities"

    for i in range(0, len(headlines), SUMMARY_BATCH_SIZE):
        batch = headlines[i:i + SUMMARY_BATCH_SIZE]
        prompt = "\n".join([
            f"Title: {h['title']}\nSnippet: {h.get('snippet','')}\nURL: {h['url']}"
            for h in batch
        ])

        try:
            resp = await client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional summarizer for {domain} {type_str}.\n"
                            "Respond with one JSON object per line like this:\n"
                            "{\"summary\": \"...\", \"expectations\": \"...\", \"highlights\": \"...\"}\n"
                            "- summary: 1 short complete sentence, proper grammar.\n"
                            "- expectations: what the user gains or learns.\n"
                            "- highlights: 2-3 short keywords separated by commas.\n"
                            "No bullets, emojis, or special characters."
                        )
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
            )

            daily_counters["ai_calls"] += 1
            ai_lines = resp.choices[0].message.content.strip().splitlines()

            for idx, h in enumerate(batch):
                snippet = clean_text(h.get("snippet", ""))
                title = clean_text(h["title"])
                summary = snippet or title
                expectations = snippet
                highlights = title.split()[:3]

                if idx < len(ai_lines):
                    line = ai_lines[idx].strip()
                    line = re.sub(r"[^\x20-\x7E]+", "", line)
                    m = re.match(
                        r'.*?"summary"\s*:\s*"([^"]+)"[^}]*"expectations"\s*:\s*"([^"]+)"[^}]*"highlights"\s*:\s*"([^"]+)"',
                        line
                    )
                    if m:
                        summary = clean_text(m.group(1))
                        expectations = clean_text(m.group(2))
                        highlights = [clean_text(x) for x in m.group(3).split(",")]

                if summary and not summary.endswith(('.', '!', '?')):
                    summary += '.'

                summarized.append({
                    "id": h["url"] or f"{domain}-{i}-{idx}",
                    "label": title,
                    "title": title,
                    "url": h["url"],
                    "summary": summary[:200],
                    "description": snippet,
                    "expectations": expectations,
                    "highlights": highlights,
                    "domain": domain,
                    "source": "brave",
                })

        except Exception as e:
            log(f"⚠️ AI summarization error: {e}")
            for idx, h in enumerate(batch):
                snippet = clean_text(h.get("snippet", ""))
                title = clean_text(h["title"])
                summarized.append({
                    "id": h["url"] or f"{domain}-{i}-{idx}",
                    "label": title,
                    "title": title,
                    "url": h["url"],
                    "summary": snippet[:200] or title,
                    "description": snippet,
                    "expectations": snippet or title,
                    "highlights": title.split()[:3],
                    "domain": domain,
                    "source": "brave",
                })

    return summarized


# === Main Flow ===
async def summarize_headline_search(query: str, domain: str, types: List[str]) -> List[Dict]:
    reset_counters_if_new_day()
    if daily_counters["calls"] >= MAX_CALLS_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily call limit reached (10 calls)")

    daily_counters["calls"] += 1
    results = await brave_search(query)
    if not results:
        return []

    for r in results:
        r["score"] = await ai_rate_result(r["title"], r.get("snippet", ""), domain, types)
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return await ai_summarize_headlines(sorted_results, domain, types)


# === FastAPI ===
app = FastAPI(title="Tech Opportunities API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://techstack.vercel.app", "https://techstack-omega.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/certifications")
async def certifications(
    domain: str,
    search: Optional[str] = "",
    level: Optional[str] = "All",
    types: str = "Certifications",
    platforms: str = "",
):
    query_parts = [domain]
    if search:
        query_parts.append(search)
    if level and level != "All":
        query_parts.append(level)
    if types:
        query_parts.append(types)
    if platforms:
        query_parts.append(platforms)

    q = " ".join(query_parts)
    type_list = types.split(",") if types else []

    return await summarize_headline_search(q, domain, type_list)


@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}
