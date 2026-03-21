import re

def extract_user_profile(text: str) -> list:
    """
    Extract user preferences/goals from text.
    """
    memories = []

    text_lower = text.lower()

    # Preference patterns
    if "i like" in text_lower or "i prefer" in text_lower:
        memories.append({
            "type": "profile",
            "category": "preference",
            "value": text
        })

    # Goals
    if "i want to" in text_lower or "my goal" in text_lower:
        memories.append({
            "type": "profile",
            "category": "goal",
            "value": text
        })

    # Skills
    if "i am learning" in text_lower or "i am studying" in text_lower:
        memories.append({
            "type": "profile",
            "category": "skill",
            "value": text
        })

    return memories