# llm_recommend.py
def heuristic_recommendation(overall_score: float, news_score: float, social_score: float, econ_score: float, headlines: list):
    if overall_score >= 0.35:
        rec = "BUY"
    elif overall_score <= -0.35:
        rec = "SELL"
    else:
        rec = "HOLD"

    reasons = []
    if news_score >= 0.2:
        reasons.append("News sentiment strongly positive.")
    if news_score <= -0.2:
        reasons.append("News sentiment strongly negative.")
    if social_score >= 0.2:
        reasons.append("Social chatter positive.")
    if social_score <= -0.2:
        reasons.append("Social chatter negative.")
    if abs(econ_score) >= 0.2:
        if econ_score > 0:
            reasons.append("Economic indicators favorable.")
        else:
            reasons.append("Economic indicators weak.")
    if not reasons:
        reasons.append("Mixed signals; consider price action & fundamentals.")

    top = headlines[0] if headlines else "No headline available."
    text = f"{rec}: {'; '.join(reasons[:3])}. Top headline: {top}. Note: Opinion — not financial advice."
    return text
