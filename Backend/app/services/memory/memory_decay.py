from datetime import datetime

def apply_decay(metadata: dict, decay_rate: float = 0.01) -> dict:
    """
    Reduce importance over time.
    """
    importance = metadata.get("importance", 0.5)
    # Support both 'timestamp' and 'created_at' for backward compatibility
    timestamp = metadata.get("timestamp") or metadata.get("created_at")

    if not timestamp:
        return metadata

    try:
        t = datetime.fromisoformat(timestamp)
        hours = (datetime.now() - t).total_seconds() / 3600

        decay = hours * decay_rate
        importance = max(0.1, importance - decay)

        metadata["importance"] = importance

    except:
        pass

    return metadata