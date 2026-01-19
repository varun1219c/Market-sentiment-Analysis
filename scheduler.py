# scheduler.py
import os, json, time
from apscheduler.schedulers.background import BackgroundScheduler
from ingest import load_sample_news, load_sample_social, load_sample_econ, fetch_news_api
from sentiment import ensemble_score, score_to_rating
from llm_recommend import heuristic_recommendation  # included below (small helper)
CACHE_FILE = "cache_snapshot.json"

def aggregate_for_instrument(instrument="XYZ", newsapi_key=None):
    # fetch data (use real APIs if you wire them up)
    news = fetch_news_api(instrument, newsapi_key) if newsapi_key else load_sample_news()
    social = load_sample_social()
    econ = load_sample_econ()

    news_scores = []
    for a in news:
        text = (a.get("title","") + " " + a.get("text","") ).strip()
        s = ensemble_score(text)
        news_scores.append({"score": s, "title": a.get("title"), "source": a.get("source"), "url": a.get("url")})

    social_scores = []
    for s in social:
        text = s.get("text","") or s.get("tweet","")
        s_score = ensemble_score(text)
        social_scores.append({"score": s_score, "text": text, "source": s.get("source")})

    econ_scores = []
    for e in econ:
        v = e.get("impact")
        if v is None:
            v = ensemble_score(e.get("text",""))
        econ_scores.append({"score": v, "desc": e.get("text",""), "published": e.get("published")})

    news_avg = sum([x["score"] for x in news_scores]) / max(1, len(news_scores))
    social_avg = sum([x["score"] for x in social_scores]) / max(1, len(social_scores))
    econ_avg = sum([x["score"] for x in econ_scores]) / max(1, len(econ_scores))
    # weights tuned for traders: news heavier, social next, econ smaller
    overall = 0.5 * news_avg + 0.35 * social_avg + 0.15 * econ_avg
    rating = score_to_rating(overall)
    headlines = [x["title"] for x in news_scores if x.get("title")]

    # use heuristic recommendation (LLM optional)
    rec = heuristic_recommendation(overall, news_avg, social_avg, econ_avg, headlines)

    snapshot = {
        "instrument": instrument,
        "timestamp": int(time.time()),
        "scores": {"overall": overall, "news": news_avg, "social": social_avg, "econ": econ_avg},
        "rating": rating,
        "top_headlines": headlines[:10],
        "recommendation": rec,
        "news_items": news_scores,
        "social_items": social_scores,
        "econ_items": econ_scores
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def start_scheduler(instrument="XYZ", newsapi_key=None, interval_seconds=20):
    sched = BackgroundScheduler()
    sched.add_job(lambda: aggregate_for_instrument(instrument, newsapi_key), "interval", seconds=interval_seconds, next_run_time=None)
    sched.start()
    # run once immediately
    aggregate_for_instrument(instrument, newsapi_key)
