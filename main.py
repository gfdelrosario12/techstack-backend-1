import asyncio
import os
import uuid
import datetime
from typing import Dict, List
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from fastapi import Request, HTTPException

# === Load Environment Variables ===
load_dotenv()

# === OpenAI Client ===
client_ai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
BING_SEARCH_URL = "https://www.bing.com/search"
SEARCH_DELAY = 2
MAX_RESULTS = 10
DAILY_LIMIT = 1

# Track API calls per day
CALL_TRACKER = {"date": datetime.date.today(), "count": 0}

# Tech domains and categories
TECH_DOMAINS = ["cloud", "ai", "developer", "cybersecurity", "data", "networking"]

OPPORTUNITY_TYPES = {
    "certifications": ["certification", "exam", "official certificate"],
    "vouchers": ["voucher", "discount", "free exam voucher"],
    "courses": ["course", "learning path", "bootcamp"],
    "events": ["event", "webinar", "hackathon"],
}

SOCIAL_PLATFORMS = [
    "linkedin",
    "reddit",
    "facebook",
    "medium",
    "dev.to",
    "devpost",
]

# === Logging Helper ===
def log(message: str):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# === Rate Limit Checker ===
def check_rate_limit() -> bool:
    today = datetime.date.today()
    if CALL_TRACKER["date"] != today:
        CALL_TRACKER["date"] = today
        CALL_TRACKER["count"] = 0
    if CALL_TRACKER["count"] >= DAILY_LIMIT:
        log("❌ Daily API call limit reached")
        return False
    CALL_TRACKER["count"] += 1
    log(f"✅ API call #{CALL_TRACKER['count']} for today")
    return True

# === Helper Classes ===
class OpportunityBot:
    def __init__(self):
        self.entries = {}

    def is_duplicate(self, title, source):
        return title in self.entries and source in self.entries[title]

    def add_entry(self, title, source):
        self.entries.setdefault(title, []).append(source)
        return True

# === Async Helpers ===
async def search_engine(query: str, source: str, client: httpx.AsyncClient) -> List[Dict]:
    """Search DuckDuckGo or Bing and return unique English results."""
    try:
        log(f"🔍 Searching {source} for: {query}")
        if source == "duckduckgo":
            response = await client.post(DUCKDUCKGO_SEARCH_URL, data={"q": query, "kl": "us-en"})
            links_selector = ".result a.result__a"
        else:
            response = await client.get(BING_SEARCH_URL, params={"q": query, "setLang": "EN"})
            links_selector = "li.b_algo h2 a"

        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select(links_selector)

        seen_urls = set()
        results = []

        for link in links[:5]:
            title = link.get_text(strip=True)
            url = link.get("href")

            if not title or not url:
                continue

            # Skip non-English titles (basic ASCII check)
            if sum(c.isascii() for c in title) / len(title) < 0.9:
                log(f"⚠️ Skipped non-English: {title}")
                continue

            if url in seen_urls:
                log(f"⚠️ Skipped duplicate URL: {url}")
                continue
            seen_urls.add(url)

            results.append({"title": title, "url": url, "source": source})

        log(f"✅ {len(results)} results from {source} for '{query}'")
        return results[:2]  # return 2 results per engine

    except Exception as e:
        log(f"❌ Search error on {source}: {e}")
        return []

async def social_media_search(query: str, platform: str, client: httpx.AsyncClient) -> List[Dict]:
    """Use DuckDuckGo to search inside a social platform."""
    site_query = f"site:{platform} {query}"
    results = await search_engine(site_query, "duckduckgo", client)
    for r in results:
        r["platform"] = platform
    return results

async def fetch_page_text(url: str, client: httpx.AsyncClient) -> str:
    """Fetch and extract clean text from a URL."""
    try:
        log(f"🌐 Fetching page: {url}")
        response = await client.get(url, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=' ', strip=True)[:1200]
    except Exception as e:
        log(f"⚠️ Failed to fetch page {url}: {e}")
        return ""

# === AI Helpers ===
async def formalize_title(title: str) -> str:
    """Use GPT to make a title more formal, skip very short titles to save tokens."""
    if not title or len(title.split()) <= 3:
        return title
    try:
        log(f"🤖 Formalizing title: {title}")
        response = await client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Make this tech title concise and formal:\n{title}"}],
            max_tokens=15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log(f"⚠️ Title AI error: {e}")
        return title

async def summarize_opportunity(text: str, label: str) -> Dict:
    """Summarize opportunity using GPT-4o-mini."""
    if not text:
        return {}
    try:
        log(f"📝 Summarizing {label} content")
        response = await client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize this {label} in 2-3 sentences:\n{text[:800]}"}],
            max_tokens=120,
        )
        generated_text = response.choices[0].message.content.strip()
        return {"summary": generated_text, "description": generated_text, "expectations": generated_text}
    except Exception as e:
        log(f"⚠️ Summary AI error: {e}")
        return {}

async def process_result(domain: str, query: str, label: str, entry: Dict, bot: OpportunityBot, client: httpx.AsyncClient):
    normalized_title = entry["title"].strip().lower()
    if bot.is_duplicate(normalized_title, entry["source"]):
        log(f"⚠️ Skipped duplicate title: {entry['title']}")
        return None

    formal_title_task = asyncio.create_task(formalize_title(entry["title"]))
    page_text_task = asyncio.create_task(fetch_page_text(entry["url"], client))
    formal_title, page_text = await asyncio.gather(formal_title_task, page_text_task)

    summary_data = await summarize_opportunity(page_text, label)
    bot.add_entry(normalized_title, entry["source"])

    log(f"✅ Processed: {formal_title}")
    return {
        "id": str(uuid.uuid4()),
        "label": domain.capitalize(),
        "title": formal_title,
        "url": entry["url"],
        "summary": summary_data.get("summary", ""),
        "description": summary_data.get("description", ""),
        "expectations": summary_data.get("expectations", ""),
        "category_label": label.capitalize(),
        "domain": domain,
        "source": entry.get("source", ""),
        "platform": entry.get("platform", ""),
        "search_query": query
    }

# === FastAPI App ===
origins = ["http://localhost:3000", "http://127.0.0.1:3000", "https://techstack.vercel.app"]
app = FastAPI(title="Tech Certifications Finder API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/certifications")
async def get_certifications(
    domain: str = Query(...),
    search: str = Query(""),
    types: List[str] = Query(["certifications"]),
):
    if not check_rate_limit():
        return JSONResponse(status_code=429, content={"error": "Daily API call limit reached (1 call/day)."})

    domain = domain.lower()
    if domain not in TECH_DOMAINS:
        return {"error": f"Invalid tech domain. Choose from {TECH_DOMAINS}"}

    types = [t.lower() for t in types if t.lower() in OPPORTUNITY_TYPES] or ["certifications"]

    bot = OpportunityBot()
    results_summary = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for opp_type in types:
                for keyword in OPPORTUNITY_TYPES[opp_type]:
                    if len(results_summary) >= MAX_RESULTS:
                        break

                    query = f"{domain} {keyword} {search}"
                    log(f"🔎 Starting search for '{query}'")

                    for platform in SOCIAL_PLATFORMS:
                        if len(results_summary) >= MAX_RESULTS:
                            break
                        results = await social_media_search(query, platform, client)
                        for entry in results:
                            if len(results_summary) >= MAX_RESULTS:
                                break
                            processed = await process_result(domain, query, opp_type, entry, bot, client)
                            if processed:
                                results_summary.append(processed)

                    await asyncio.sleep(SEARCH_DELAY)

        log(f"🎯 Completed: {len(results_summary)} results returned")
        return {"count": len(results_summary), "results": results_summary}

    except Exception as e:
        log(f"❌ Server error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset-limit")
async def reset_limit(request: Request):
    client_ip = request.client.host
    if client_ip not in ["127.0.0.1", "::1"]:
        raise HTTPException(status_code=403, detail="Access forbidden")

    CALL_TRACKER["date"] = datetime.date.today()
    CALL_TRACKER["count"] = 0
    return {"message": f"Daily limit reset for {CALL_TRACKER['date']}"}

@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}