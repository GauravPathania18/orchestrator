"""
Memory extraction module - Abhishek's Week 2 Implementation.

Provides two approaches:
1. Regex extraction (fallback) - Fast, rule-based fragment extraction
2. LLM extraction (preferred) - Uses LLM to intelligently extract structured facts

The main entry point is extract_user_profile_llm() which tries LLM first,
then falls back to regex extraction if needed.
"""
import re
import json
import logging
from datetime import datetime
from typing import Optional

from Backend.app.services.ollama_client import generate_response

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_PROMPT = """You are a memory extraction system.
Given a user message, extract ONLY stable, useful facts about the user.

Rules:
- Extract fragments, not full sentences
- Ignore questions the user is asking
- Ignore temporary states ("I'm tired")
- Return JSON only, no explanation

Output format:
{
  "memories": [
    {"category": "preference", "value": "Python"},
    {"category": "skill",      "value": "learning Rust"}
  ]
}

If nothing useful found: {"memories": []}

User message: "{message}"
"""


async def extract_memories_with_llm(message: str, confidence_threshold: float = 0.8) -> list[dict]:
    """
    Extract memories from a message using LLM-based extraction.
    This is the preferred method (Approach 2) - Abhishek's Week 2 implementation.

    Args:
        message: The user message to extract memories from
        confidence_threshold: Minimum confidence score for accepting memories

    Returns:
        List of memory dicts with category, value, confidence, and timestamp
    """
    prompt = MEMORY_EXTRACTION_PROMPT.format(message=message)

    try:
        response = await generate_response(
            prompt=prompt,
            temperature=0.1  # Low temperature for structured output
        )

        # Parse JSON response
        try:
            # Clean up response to handle potential markdown code blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            data = json.loads(cleaned_response)
            memories = data.get("memories", [])

            # Add confidence and timestamp to each memory
            enriched_memories = []
            for memory in memories:
                if "category" in memory and "value" in memory:
                    enriched_memories.append({
                        "type": "profile",
                        "category": memory["category"],
                        "value": memory["value"],
                        "confidence": confidence_threshold,
                        "timestamp": datetime.now().isoformat()
                    })

            return enriched_memories

        except json.JSONDecodeError:
            logger.warning(f"Memory extraction parse failed for: {message}")
            logger.debug(f"Raw response: {response}")
            return []

    except Exception as e:
        logger.error(f"LLM memory extraction failed: {e}")
        return []


async def extract_user_profile_llm(text: str, use_regex_fallback: bool = True) -> list:
    """
    Extract user profile using LLM-based extraction with optional regex fallback.

    This is the main entry point for memory extraction (Abhishek's Week 2).
    Falls back to regex extraction (Approach 1) if LLM fails or returns empty.

    Args:
        text: User message text
        use_regex_fallback: Whether to fall back to regex extraction if LLM fails

    Returns:
        List of memory dictionaries
    """
    # Try LLM extraction first (Approach 2)
    memories = await extract_memories_with_llm(text)

    # Fallback to regex if LLM returns nothing and fallback is enabled
    if not memories and use_regex_fallback:
        logger.debug("LLM extraction returned empty, using regex fallback")
        memories = extract_user_profile_regex(text)

    return memories


def extract_preference_fragments(text: str) -> list[str]:
    """
    Pull only the preference fragment from a message,
    not the entire raw text.
    """
    text_lower = text.lower()
    fragments = []

    # Pattern: capture what comes after "i like/prefer/enjoy/love" (with optional "also")
    # until a clause boundary (but, though, what, and, comma, period, etc.)
    like_pattern = re.findall(
        r"i (?:also )?(?:like|love|prefer|enjoy)\s+([^\.\?!]+?)(?:\s*,|\s*but|\s*though|\s*what|\s*and|\.|\?|!|$)",
        text_lower
    )
    fragments.extend(like_pattern)

    # Clean and normalize - take only first few words to avoid capturing noise
    cleaned = []
    for f in fragments:
        f = f.strip()
        if len(f) > 1:
            # Limit to first 4 words to keep it clean
            words = f.split()[:4]
            cleaned.append(' '.join(words))
    return cleaned


def extract_goal_fragments(text: str) -> list[str]:
    """
    Extract goal fragments from a message.
    """
    text_lower = text.lower()
    fragments = []

    # Pattern: capture what comes after "i want to" or "my goal is"
    want_pattern = re.findall(
        r"i want to\s+([^,\.?!]+?)(?:\s+but|\s+though|\s+what|\s+because|$)",
        text_lower
    )
    goal_pattern = re.findall(
        r"my goal (?:is|was)?\s+(?:to\s+)?([^,\.?!]+?)(?:\s+but|\s+though|\s*what|$)",
        text_lower
    )
    fragments.extend(want_pattern)
    fragments.extend(goal_pattern)

    return [f.strip() for f in fragments if len(f.strip()) > 1]


def extract_skill_fragments(text: str) -> list[str]:
    """
    Extract skill/learning fragments from a message.
    """
    text_lower = text.lower()
    fragments = []

    # Pattern: capture what comes after "i am learning/studying"
    learning_pattern = re.findall(
        r"(?:i'm|i am)\s+(?:also\s+)?(?:learning|studying)\s+([a-zA-Z0-9\+\#\s]+?)(?:\s+but|\s+though|\s+what|\s+and|,|\.|$)",
        text_lower
    )
    fragments.extend(learning_pattern)

    return [f.strip() for f in fragments if len(f.strip()) > 1]


def extract_user_profile_regex(text: str) -> list:
    """
    Extract user preferences/goals from text using regex (Approach 1 - Fallback).
    """
    memories = []

    text_lower = text.lower()

    # Preference patterns - extract fragments, not full text
    if any(pattern in text_lower for pattern in ["i like", "i also like", "i prefer", "i love", "i enjoy"]):
        fragments = extract_preference_fragments(text)
        for fragment in fragments:
            memories.append({
                "type": "profile",
                "category": "preference",
                "value": fragment
            })

    # Goals - extract fragments
    if "i want to" in text_lower or "my goal" in text_lower:
        fragments = extract_goal_fragments(text)
        for fragment in fragments:
            memories.append({
                "type": "profile",
                "category": "goal",
                "value": fragment
            })

    # Skills - extract fragments
    if "i am learning" in text_lower or "i'm learning" in text_lower or "i am studying" in text_lower or "i'm studying" in text_lower:
        fragments = extract_skill_fragments(text)
        for fragment in fragments:
            memories.append({
                "type": "profile",
                "category": "skill",
                "value": fragment
            })

    return memories


# Alias for backward compatibility
extract_user_profile = extract_user_profile_regex
