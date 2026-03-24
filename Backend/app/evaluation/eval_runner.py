import asyncio

from app.services.rag_adapter import (
    raptor_retrieve,
    baseline_retrieve,
    generate_answer
)
from app.evaluation.eval_datasets import (
    load_quasper,
    load_narrativeqa,
    load_quality,
    prepare_quasper,
    prepare_narrative,
)
from app.evaluation.metrics import compute_f1
from app.evaluation.quality_test import evaluate_quality


async def run_pipeline(question, mode="raptor"):

    if mode == "baseline":
        retrieved = await baseline_retrieve(question)
    else:
        retrieved = await raptor_retrieve(question)

    context = "\n".join([c["text"] for c in retrieved])

    response = await generate_answer(question, context)

    return response, retrieved

async def evaluate_quasper(dataset, mode="raptor"):
    scores = []

    for sample in dataset:
        question, _, gt = prepare_quasper(sample)

        response, _ = await run_pipeline(question, mode)

        score = compute_f1(response, gt)
        scores.append(score)

    print(f"QUASPER ({mode}) F1:", sum(scores) / len(scores))


async def evaluate_narrative(dataset, mode="raptor"):
    scores = []

    for sample in dataset:
        question, _, gt = prepare_narrative(sample)

        response, _ = await run_pipeline(question, mode)

        score = compute_f1(response, gt)
        scores.append(score)

    print(f"NarrativeQA ({mode}) F1:", sum(scores) / len(scores))


async def main():

    print("Loading datasets...")

    quasper = load_quasper()
    narrative = load_narrativeqa()
    quality = load_quality()

    print("\n--- QUALITY TEST ---")
    await evaluate_quality(run_pipeline, quality)

    print("\n--- QUASPER TEST ---")
    await evaluate_quasper(quasper, mode="baseline")
    await evaluate_quasper(quasper, mode="raptor")

    print("\n--- NARRATIVE QA TEST ---")
    await evaluate_narrative(narrative, mode="baseline")
    await evaluate_narrative(narrative, mode="raptor")


if __name__ == "__main__":
    asyncio.run(main())