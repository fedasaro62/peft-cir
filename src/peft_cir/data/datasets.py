"""Torch datasets and per-benchmark train/validate plumbing.

Triplets are driven by the :mod:`peft_cir.data.benchmarks` loaders. Validation runs through
:func:`peft_cir.evaluation.ranked_eval`, the same ranking code as the reported evaluation, so a
checkpoint-selection score can never drift from the published number.

CIRCO is absent from the train side on purpose -- it ships no train split.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from peft_cir.data import metrics
from peft_cir.data.benchmarks import (
    VAL_FRACTION,
    Benchmark,
    BenchQuery,
    build_cirr,
    build_fiq,
    build_shoes,
)
from peft_cir.evaluation import ranked_eval

FIQ_SUBTASKS = ["dress", "shirt", "toptee"]

# per-benchmark checkpoint-selection keys added to the metrics dict
SELECTION_KEYS = {
    "cirr": "cirr_avg",           # (Recall@5 + Recall_subset@1) / 2
    "fiq": "fiq_avg",             # mean over subtasks of (Recall@10 + Recall@50) / 2
    "shoes": "shoes_avg",         # (Recall@10 + Recall@50) / 2
}

Tokenizer = Callable[[list[str]], torch.Tensor]


class BenchTripletDataset(Dataset):
    """Serves (reference pixels, target pixels, prompt tokens) for benchmark triplets."""

    def __init__(self, queries: Sequence[BenchQuery], id_to_path: dict[Any, str],
                 preprocess: Any, tokens: torch.Tensor) -> None:
        self.queries, self.id_to_path = list(queries), id_to_path
        self.preprocess, self.tokens = preprocess, tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def _pixels(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.preprocess(images=img, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.queries[i]
        return self._pixels(q.ref_path), self._pixels(self.id_to_path[q.target]), self.tokens[i]


def index_path_map(*benches: Benchmark) -> dict[Any, str]:
    """index id -> image path across one or more benchmarks, for resolving target images.

    Merging is safe across FashionIQ subtasks: shirt and toptee share image ids, but the
    path is derived from the id alone, so a shared key always maps to the same path.
    """
    merged: dict[Any, str] = {}
    for b in benches:
        merged.update(zip(b.index_ids, b.index_paths))
    return merged


def _dataset(bench: Benchmark, preprocess: Any, tokenize: Tokenizer) -> BenchTripletDataset:
    return BenchTripletDataset(bench.queries, index_path_map(bench), preprocess,
                               tokenize([q.caption for q in bench.queries]))


def build_train(dataset: str, data_root: str | Path, preprocess: Any, tokenize: Tokenizer,
                val_frac: float = VAL_FRACTION) -> BenchTripletDataset:
    """The train split of a supported benchmark, as a triplet dataset.

    FashionIQ pools its three subtasks into one jointly trained model; Shoes carves val out of
    train (``val_frac``).
    """
    if dataset == "fiq":
        benches = [build_fiq(data_root, sub, "train") for sub in FIQ_SUBTASKS]
        queries = [q for b in benches for q in b.queries]
        return BenchTripletDataset(queries, index_path_map(*benches), preprocess,
                                   tokenize([q.caption for q in queries]))
    if dataset == "cirr":
        return _dataset(build_cirr(data_root, "train"), preprocess, tokenize)
    if dataset == "shoes":
        return _dataset(build_shoes(data_root, "train", val_frac=val_frac), preprocess, tokenize)
    raise ValueError(f"no train split for {dataset!r} (want one of {sorted(SELECTION_KEYS)})")


def build_val(dataset: str, data_root: str | Path,
              split: str = "val") -> Benchmark | list[Benchmark]:
    """The benchmark(s) validation ranks against; FashionIQ returns one per subtask."""
    if dataset == "fiq":
        return [build_fiq(data_root, sub, split) for sub in FIQ_SUBTASKS]
    if dataset == "cirr":
        return build_cirr(data_root, split)
    if dataset == "shoes":
        return build_shoes(data_root, split)
    raise ValueError(f"unknown benchmark {dataset!r}")


def avg_r10_r50(m: dict[str, float]) -> float:
    """FashionIQ's "Avg", and the same summary Shoes is selected on."""
    return (m["Recall@10"] + m["Recall@50"]) / 2.0


def cirr_avg(m: dict[str, float]) -> float:
    """The standard CIRR summary number: mean of global Recall@5 and subset Recall@1."""
    return (m["Recall@5"] + m["Recall_subset@1"]) / 2.0


def validate(dataset: str, model: Any, preprocess: Any, bench, device: str,
             batch_size: int) -> dict[str, float]:
    """Validation metrics for the current model state, plus this benchmark's selection key."""
    if dataset == "fiq":
        per_sub = {b.name.replace("fashioniq-", ""):
                   ranked_eval(model, preprocess, b, device, batch_size, metrics.fiq_recall)
                   for b in bench}
        out = {f"{sub}_{k}": v for sub, ms in per_sub.items() for k, v in ms.items()}
        for k in ("Recall@10", "Recall@50"):
            out[k] = sum(ms[k] for ms in per_sub.values()) / len(per_sub)
        out[SELECTION_KEYS["fiq"]] = avg_r10_r50(out)
        return out

    if dataset == "cirr":
        out = ranked_eval(model, preprocess, bench, device, batch_size, metrics.cirr_metrics)
        out[SELECTION_KEYS["cirr"]] = cirr_avg(out)
        return out

    out = ranked_eval(model, preprocess, bench, device, batch_size, metrics.shoes_recall)
    out[SELECTION_KEYS["shoes"]] = avg_r10_r50(out)
    return out
