from ..utils import cosine_similarity

def select_top_memories(memories: list, top_k: int = 5, similarity_threshold: float = 0.85):

    if not memories:
        return []

    memories = sorted(memories, key=lambda x: x.get("score", 0), reverse=True)

    selected = []
    selected_embeddings = []
    seen_texts = set()

    profile_count = 0
    MAX_PROFILE = 2

    for m in memories:
        doc = (m.get("document") or "").strip()
        meta = m.get("metadata", {})
        emb = m.get("embedding")
        score = m.get("score", 0)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        if not doc or score <= 0:
            continue

        doc_key = doc.lower()

        if doc_key in seen_texts:
            continue

        # -----------------------------
        # DIVERSITY FILTER
        # -----------------------------
        if emb is not None and selected_embeddings:
            if any(cosine_similarity(emb, e) > similarity_threshold for e in selected_embeddings):
                continue

        # -----------------------------
        # PROFILE PRIORITY
        # -----------------------------
        if meta.get("type") == "profile":
            if profile_count < MAX_PROFILE:
                selected.append(m)
                seen_texts.add(doc_key)
                if emb is not None:
                    selected_embeddings.append(emb)
                profile_count += 1
            continue

        # -----------------------------
        # NORMAL
        # -----------------------------
        selected.append(m)
        seen_texts.add(doc_key)

        if emb is not None:
            selected_embeddings.append(emb)

        if len(selected) >= top_k:
            break

    # -----------------------------
    # ENSURE PROFILE PRESENCE
    # -----------------------------
    if profile_count == 0:
        for m in memories:
            if m.get("metadata", {}).get("type") == "profile":
                selected.insert(0, m)
                selected = selected[:top_k]
                break

    return selected[:top_k]