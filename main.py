from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

app = FastAPI()
@app.get("/")
def home():
    return {"message": "TruthTrace API is running"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------- Request Models --------

class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str


# -------- Extract Article Text + Source Name --------

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract article text
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        # Extract source name from meta tag
        site_name = None
        meta_tag = soup.find("meta", property="og:site_name")
        if meta_tag:
            site_name = meta_tag.get("content")

        # Fallback to domain name
        if not site_name:
            parsed_url = urlparse(url)
            site_name = parsed_url.netloc.replace("www.", "")

        return article_text[:5000], site_name

    except Exception:
        return None, None


# -------- AI Fact Check Function --------

def fact_check(content):
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Current date: {current_date}. "
                    "You are a strict fact-checking AI. "
                    "Use the current date above as today's date. "
                    "Respond ONLY in valid JSON format like this:\n"
                    "{\n"
                    '  "verdict": "Real or Fake",\n'
                    '  "reason": "Clear explanation",\n'
                    '  "confidence_percent": number (0-100)\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    output = response.choices[0].message.content.strip()

    try:
        return json.loads(output)
    except:
        return {
            "verdict": "Unclear",
            "reason": output,
            "confidence_percent": 0,
        }
# -------- Text Endpoint --------

@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    try:
        return fact_check(request.text)
    except Exception as e:
        return {"error": str(e)}


# -------- URL Endpoint --------

@app.post("/analyze-url")
def analyze_url(request: URLRequest):
    try:
        article_text, source_name = extract_text_from_url(request.url)

        if not article_text:
            return {"error": "Could not extract article content."}

        result = fact_check(article_text)
        result["source_name"] = source_name

        return result

    except Exception as e:
        return {"error": str(e)}

