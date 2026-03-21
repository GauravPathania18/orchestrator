from datetime import datetime

def compute_memory_score(metadata: dict, similarity: float) -> float:

    importance = metadata.get("importance", 0.5)
    confidence = metadata.get("confidence", 0.5)

    # -----------------------------
    # RECENCY
    # -----------------------------
    recency = 1.0
    ts = metadata.get("last_accessed")

    if ts:
        try:
            t = datetime.fromisoformat(ts)
            hours = (datetime.now() - t).total_seconds() / 3600

            # smoother decay curve
            recency = max(0.6, 1 / (1 + hours / 24))

        except:
            pass

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    # 🔥 balanced semantic scoring
    score = (similarity ** 1.5) * importance * confidence * recency

    return score