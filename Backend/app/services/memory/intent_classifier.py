def classify_intent(query: str) -> str:
    q = query.lower()

    if any(word in q for word in ["remember", "i like", "i prefer", "my goal"]):
        return "memory_write"

    if any(word in q for word in ["what do you know about me", "my preferences", "my goal"]):
        return "memory_read"

    if any(word in q for word in ["who", "what", "how", "why", "explain"]):
        return "knowledge"

    return "general"