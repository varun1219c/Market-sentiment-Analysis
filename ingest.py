# ingest.py
import os, json
from typing import List, Dict

SAMPLE_DIR = "sample_data"

def _load_json(name):
    p = os.path.join(SAMPLE_DIR, name)
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sample_news() -> List[Dict]:
    return _load_json("news.json")

def load_sample_social() -> List[Dict]:
    return _load_json("social.json")

def load_sample_econ() -> List[Dict]:
    return _load_json("econ.json")

# TODO: replace with real APIs (NewsAPI, Tweepy) when you have keys
def fetch_news_api(query: str, api_key: str):
    # placeholder: return sample news if no key
    return load_sample_news()
