def reinforce_memory(metadata: dict, boost: float = 0.05) -> dict:
    """
    Increase importance of useful memory.
    """
    importance = metadata.get("importance", 0.5)
    importance = min(1.0, importance + boost)

    metadata["importance"] = importance
    return metadata