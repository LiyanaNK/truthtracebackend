from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def home():
    return {"message": "TruthTrace API is running"}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Request Models --------
class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

# -------- Extract Article Text --------
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        # Get site name
        site_name = None
        meta_tag = soup.find("meta", property="og:site_name")
        if meta_tag:
            site_name = meta_tag.get("content")

        if not site_name:
            parsed_url = urlparse(url)
            site_name = parsed_url.netloc.replace("www.", "")

        return article_text[:3000], site_name   # 🔥 reduced for speed

    except Exception:
        return None, None

# -------- Search Related News --------
def search_related_news(query):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": " ".join(query.split()[:10]),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 3,   # 🔥 faster
            "apiKey": os.getenv("NEWS_API_KEY")
        }

        response = requests.get(url, params=params)
        data = response.json()

        headlines = []
        for article in data.get("articles", []):
            headlines.append(article["title"])

        return headlines

    except:
        return []

# -------- AI Fact Check --------
def fact_check(content):
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        related_news = search_related_news(content)
        context = "\n".join(related_news)

        prompt = f"""
Current date: {current_date}

You are a real-time fact checking AI.

Claim:
{content}

Related recent news:
{context}

IMPORTANT:
Return ONLY pure JSON.
Do NOT include markdown like ```json.

Format:
{{
    "verdict": "Real, Fake, or Unverified",
    "reason": "Explanation",
    "confidence_percent": number (0-100)
}}
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        output = response.text.strip()

        # 🔥 Remove markdown if Gemini adds it
        if "```" in output:
            output = output.replace("```json", "").replace("```", "").strip()

        return json.loads(output)

    except Exception as e:
        return {
            "verdict": "Error",
            "reason": str(e),
            "confidence_percent": 0
        }

# -------- Text Endpoint --------
@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    return fact_check(request.text)

# -------- URL Endpoint --------
@app.post("/analyze-url")
def analyze_url(request: URLRequest):
    article_text, source_name = extract_text_from_url(request.url)

    if not article_text:
        return {"error": "Could not extract article content."}

    result = fact_check(article_text)
    result["source_name"] = source_name

    return result