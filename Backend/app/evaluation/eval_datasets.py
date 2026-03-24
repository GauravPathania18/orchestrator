from datasets import load_dataset


# -----------------------------
# SQUAD (REPLACES QUASPER)
# -----------------------------
def load_quasper():
    dataset = load_dataset("squad", split="validation", streaming=True)
    return list(dataset.take(20))


# -----------------------------
# NARRATIVE QA
# -----------------------------
def load_narrativeqa():
    dataset = load_dataset("narrativeqa", split="validation", streaming=True)
    return list(dataset.take(20))  # only 20 samples


# -----------------------------
# FORMATTERS
# -----------------------------
def prepare_quasper(sample):
    question = sample["question"]
    context = sample["context"]

    answers = sample.get("answers", {}).get("text", [])
    answer = answers[0] if answers else ""

    return question, context, answer


def prepare_narrative(sample):
    question = sample["question"]["text"]
    context = sample["document"]["summary"]["text"]

    answers = sample.get("answers", [])
    answer = answers[0]["text"] if answers else ""

    return question, context, answer


# -----------------------------
# QUALITY TEST
# -----------------------------
def load_quality():
    return [
        {
            "question": "What is token-based flushing?",
            "expected_chunk": "summarize"
        },
        {
            "question": "What is retrieval augmented generation?",
            "expected_chunk": "retrieval"
        }
    ]