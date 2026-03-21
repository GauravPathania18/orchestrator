def build_structured_context(
    user_profile: list,
    short_term: list,
    knowledge: list,
    query: str
) -> str:

    def format_block(title, items):
        if not items:
            return ""
        return f"[{title}]\n" + "\n".join(f"- {i}" for i in items) + "\n\n"

    context = ""

    context += format_block("USER PROFILE", user_profile)
    context += format_block("RECENT CONVERSATION", short_term)
    context += format_block("RELEVANT KNOWLEDGE", knowledge)

    context += f"[USER QUESTION]\n{query}"

    return context