import re


def _clean(text: str) -> str:
    return re.sub(r"[\x00-\x1f]", "\n", text).strip().lower()


def score_prediction(pred: str, refs, metric: str) -> float:
    pred = _clean(pred)
    refs = [str(r).lower() for r in refs]
    if not refs:
        return 0.0
    hits = [1.0 if r in pred else 0.0 for r in refs]
    if metric == "part":
        return max(hits)
    if metric == "all":
        return sum(hits) / len(hits)
    raise ValueError(metric)


def metric_for_task(task: str) -> str:
    if task in {"qa_1", "qa_2"}:
        return "part"
    if task in {"fwe", "niah_multivalue"}:
        return "all"
    raise ValueError(f"Unsupported task for this first experiment: {task}")


def score_records(records, task: str) -> float:
    metric = metric_for_task(task)
    vals = [score_prediction(r.get("pred", r.get("generation", "")), r.get("outputs", []), metric) for r in records]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0
