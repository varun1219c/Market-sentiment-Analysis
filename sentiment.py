# sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_analyzer = SentimentIntensityAnalyzer()

# optional transformer support (will be used only if installed)
try:
    from transformers import pipeline
    _clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    TRANSFORMER_AVAILABLE = True
except Exception:
    _clf = None
    TRANSFORMER_AVAILABLE = False

def lexicon_score(text: str) -> float:
    if not text:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]  # -1..1

def transformer_score(text: str) -> float:
    if not TRANSFORMER_AVAILABLE or not text:
        return 0.0
    try:
        res = _clf(text[:512])[0]
        return res["score"] if res["label"].upper().startswith("POS") else -res["score"]
    except Exception:
        return 0.0

def ensemble_score(text: str, w_lex=0.6, w_trans=0.4) -> float:
    # default heavier weight on VADER for speed/robustness
    lx = lexicon_score(text)
    tr = transformer_score(text)
    return w_lex * lx + w_trans * tr

def score_to_rating(overall: float) -> int:
    rating = round(((overall + 1) / 2) * 4 + 1)
    return max(1, min(5, rating))
