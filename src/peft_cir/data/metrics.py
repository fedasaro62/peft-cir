"""Retrieval metrics: pure functions over per-query ranked id lists.

Kept apart from the loaders so the same scoring code serves validation inside the training
loop and the reported evaluation -- a selection score can then never drift from the number
that gets published.
"""

from collections.abc import Sequence

K_VALUES = (1, 5, 10, 50)


def _recall(ranked_per_query: Sequence[Sequence], targets: Sequence, ks: Sequence[int]) -> dict:
    hits = {k: 0 for k in ks}
    n = len(ranked_per_query)
    for ranked, tgt in zip(ranked_per_query, targets):
        rank = next((i for i, x in enumerate(ranked) if x == tgt), None)
        if rank is None:
            continue
        for k in ks:
            if rank < k:
                hits[k] += 1
    return {k: hits[k] / n * 100.0 for k in ks}


def _recall_at(ranked, queries, ks: Sequence[int]) -> dict:
    return {f"Recall@{k}": v for k, v in _recall(ranked, [q.target for q in queries], ks).items()}


def cirr_metrics(global_ranked: Sequence[Sequence[str]], subset_ranked: Sequence[Sequence[str]],
                 queries) -> dict:
    """Global Recall@{1,5,10,50} plus subset Recall@{1,2,3}, CIRR's two reported families."""
    out = _recall_at(global_ranked, queries, K_VALUES)
    out.update({f"Recall_subset@{k}": v
                for k, v in _recall(subset_ranked, [q.target for q in queries], (1, 2, 3)).items()})
    return out


def fiq_recall(global_ranked, queries) -> dict:
    return _recall_at(global_ranked, queries, (10, 50))


def shoes_recall(global_ranked, queries) -> dict:
    return _recall_at(global_ranked, queries, (1, 10, 50))


def circo_map(global_ranked: Sequence[Sequence[int]], queries,
              ks: Sequence[int] = (5, 10, 25, 50)) -> dict:
    """mAP@k over CIRCO's multiple ground-truth targets per query."""
    n = len(queries)
    out = {k: 0.0 for k in ks}
    for ranked, q in zip(global_ranked, queries):
        gt = set(q.target)
        for k in ks:
            hits, ap = 0, 0.0
            for i, x in enumerate(ranked[:k]):
                if x in gt:
                    hits += 1
                    ap += hits / (i + 1)
            out[k] += ap / min(k, len(gt))
    return {f"mAP@{k}": out[k] / n * 100.0 for k in ks}
