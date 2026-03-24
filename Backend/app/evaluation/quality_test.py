async def evaluate_quality(run_pipeline, dataset):
    results = []

    for sample in dataset:
        question = sample["question"]
        expected = sample["expected_chunk"]

        _, retrieved = await run_pipeline(question)

        texts = [c["text"] for c in retrieved]

        hit = any(expected.lower() in t.lower() for t in texts)

        results.append(hit)

    print("QUALITY Score:", sum(results) / len(results))