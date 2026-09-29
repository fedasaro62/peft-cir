"""Ranking a benchmark with a model -- the one code path both validation and reporting use.

Extracting this from the evaluation entry point is what lets
:mod:`peft_cir.data.datasets` validate mid-training with exactly the ranking code that
produces the published numbers, without the two modules importing each other.
"""

from collections.abc import Sequence

import torch
from PIL import Image
from tqdm import tqdm

from peft_cir.data.benchmarks import Benchmark, BenchQuery


@torch.no_grad()
def encode_index(model, preprocess, paths: Sequence[str], device: str, bs: int) -> torch.Tensor:
    """Candidate vectors for a gallery, in the given order."""
    feats = []
    for i in tqdm(range(0, len(paths), bs), desc="index"):
        pil = [Image.open(p).convert("RGB") for p in paths[i: i + bs]]
        pv = preprocess(images=pil, return_tensors="pt")["pixel_values"].to(device)
        feats.append(model.encode_candidates(pv).float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_queries(model, preprocess, queries: Sequence[BenchQuery], device: str,
                   bs: int) -> torch.Tensor:
    """Composed query vectors. Each arch tokenizes its own prompt, so tokenizing is its job."""
    feats = []
    for i in tqdm(range(0, len(queries), bs), desc="queries"):
        batch = queries[i: i + bs]
        pil = [Image.open(q.ref_path).convert("RGB") for q in batch]
        pv = preprocess(images=pil, return_tensors="pt")["pixel_values"].to(device)
        ref_feats = model.image_features(pv)
        tokens = model.tokenize([q.caption for q in batch]).to(device)
        feats.append(model.encode_query(tokens, ref_feats).float().cpu())
    return torch.cat(feats, dim=0)


def _topk_ranked(scores: torch.Tensor, index_ids: Sequence, k: int) -> list[list]:
    idx = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
    return [[index_ids[j] for j in row.tolist()] for row in idx]


@torch.no_grad()
def rank_benchmark(model, preprocess, bench: Benchmark, device: str, bs: int,
                   k: int = 50) -> tuple[list[list], list[list] | None]:
    """Rank the index for every query.

    CIRR additionally masks the reference out of the gallery and ranks its ``img_set`` subset,
    which is the second family of numbers that benchmark reports; every other benchmark gets
    ``None`` for the subset ranking.
    """
    index_feats = encode_index(model, preprocess, bench.index_paths, device, bs)
    q_feats = encode_queries(model, preprocess, bench.queries, device, bs)
    col = {iid: i for i, iid in enumerate(bench.index_ids)}
    scores = q_feats @ index_feats.T  # cosine (both L2-normalized)

    subset_ranked = None
    if bench.name == "cirr":
        for row, q in enumerate(bench.queries):
            if q.ref_id in col:
                scores[row, col[q.ref_id]] = float("-inf")  # mask reference from the gallery
        subset_ranked = [
            sorted((m for m in q.subset if m != q.ref_id and m in col),
                   key=lambda m: scores[row, col[m]].item(), reverse=True)
            for row, q in enumerate(bench.queries)
        ]
    return _topk_ranked(scores, bench.index_ids, k), subset_ranked


def ranked_eval(model, preprocess, bench: Benchmark, device: str, bs: int,
                metric_fn, k: int = 50) -> dict[str, float]:
    """Rank ``bench`` and score it with ``metric_fn``, restoring the model's training mode.

    Used by the in-training validators, which must not leave the model in ``eval`` afterwards.
    """
    was_training = model.training
    model.eval()
    ranked, subset = rank_benchmark(model, preprocess, bench, device, bs, k=k)
    if was_training:
        model.train()
    if bench.name == "cirr":
        return metric_fn(ranked, subset, bench.queries)
    return metric_fn(ranked, bench.queries)
