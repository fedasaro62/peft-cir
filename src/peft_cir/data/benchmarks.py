"""Dataset loaders: every benchmark reduced to the same (index, queries) shape.

A :class:`Benchmark` is what the rest of the codebase sees of a dataset -- a gallery of
image paths and a list of :class:`BenchQuery`. Scoring lives in :mod:`peft_cir.data.metrics`.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchQuery:
    qid: str
    ref_path: str                       # reference image path (composed-query image)
    caption: str                        # relative caption / modification text
    target: str | int | list[int]  # single id (cirr/fiq) or gt id list (circo)
    ref_id: str | int | None = None      # reference id in the index (for masking)
    subset: list[str] | None = None        # cirr img_set members (subset recall)


@dataclass
class Benchmark:
    name: str
    index_ids: list[str | int]
    index_paths: list[str]
    queries: list[BenchQuery]


def build_cirr(data_root: str | Path, split: str = "val") -> Benchmark:
    root = Path(data_root) / "cirr"
    tok = "test1" if split == "test" else split  # CIRR names its held-out split "test1"
    caps = json.loads((root / "annotations" / "captions" / f"cap.rc2.{tok}.json").read_text())
    imsplit = json.loads((root / "annotations" / "image_splits" / f"split.rc2.{tok}.json").read_text())

    def path_of(name: str) -> str:
        return str(root / "images" / imsplit[name].lstrip("./"))

    index_ids = list(imsplit.keys())
    index_paths = [path_of(n) for n in index_ids]
    queries = [
        BenchQuery(
            qid=str(c["pairid"]), ref_path=path_of(c["reference"]), caption=c["caption"],
            target=c.get("target_hard"), ref_id=c["reference"], subset=c["img_set"]["members"],
        )
        for c in caps
    ]
    return Benchmark("cirr", index_ids, index_paths, queries)


def build_circo(data_root: str | Path, split: str = "val") -> Benchmark:
    root = Path(data_root) / "circo"
    ann = json.loads((root / "annotations" / f"{split}.json").read_text())
    info = json.loads((root / "annotations" / "image_info_unlabeled2017.json").read_text())
    img_dir = root / "COCO2017_unlabeled" / "unlabeled2017"

    def name(i: int) -> str:
        return f"{int(i):012d}.jpg"

    index_ids = [im["id"] for im in info["images"]]
    index_paths = [str(img_dir / name(i)) for i in index_ids]
    queries = [
        BenchQuery(
            qid=str(q["id"]), ref_path=str(img_dir / name(q["reference_img_id"])),
            caption=q["relative_caption"],
            target=list(q["gt_img_ids"]) if "gt_img_ids" in q else None,
            ref_id=q["reference_img_id"],
        )
        for q in ann
    ]
    return Benchmark("circo", index_ids, index_paths, queries)


def build_fiq(data_root: str | Path, subtask: str, split: str = "val") -> Benchmark:
    root = Path(data_root) / "fashioniq"
    caps = json.loads((root / "captions" / f"cap.{subtask}.{split}.json").read_text())
    imsplit = json.loads((root / "image_splits" / f"split.{subtask}.{split}.json").read_text())

    def path_of(i: str) -> str:
        return str(root / "images" / f"{i}.png")

    index_ids = list(imsplit)
    index_paths = [path_of(i) for i in index_ids]
    queries = [
        BenchQuery(
            qid=c["candidate"], ref_path=path_of(c["candidate"]),
            caption=" and ".join(c["captions"]), target=c.get("target"), ref_id=c["candidate"],
        )
        for c in caps
    ]
    return Benchmark(f"fashioniq-{subtask}", index_ids, index_paths, queries)


def write_cirr_submission(out_dir: str | Path, global_ranked, subset_ranked, queries) -> None:
    """CIRR eval-server format: a global-recall file and a subset-recall file (rc2)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rec = {"version": "rc2", "metric": "recall"}
    sub = {"version": "rc2", "metric": "recall_subset"}
    for g, sr, q in zip(global_ranked, subset_ranked, queries):
        rec[str(q.qid)] = list(g[:50])
        sub[str(q.qid)] = list(sr[:3])
    (Path(out_dir) / "cirr_test_recall.json").write_text(json.dumps(rec))
    (Path(out_dir) / "cirr_test_subset.json").write_text(json.dumps(sub))


def write_ranked_submission(out_dir: str | Path, name: str, global_ranked, queries, k: int = 50) -> None:
    """Generic {query_id: [top-k ranked candidate ids]} submission file."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = {str(q.qid): list(g[:k]) for g, q in zip(global_ranked, queries)}
    (Path(out_dir) / f"{name}_test.json").write_text(json.dumps(out))


# --- Shoes: a benchmark with no official val split --------------------------------
#
# It ships train/test only, so `val` is carved out of train with a seeded,
# image-disjoint split (whole connected components of the reference/target graph stay
# on one side). The carve lives here so every consumer -- trainer, validator, and
# evaluator -- derives the identical split from the same seed.

VAL_FRACTION = 0.1
SPLIT_SEED = 42


def carve_val(queries: Sequence[BenchQuery], seed: int = SPLIT_SEED,
              val_frac: float = VAL_FRACTION) -> tuple[list[BenchQuery], list[BenchQuery]]:
    """Split queries into (train, val) sharing no image between the two sides.

    Queries are grouped into connected components over their reference/target images;
    whole components are assigned, so no image can appear in both sides. Components are
    shuffled with ``seed`` and accumulated into val until ``val_frac`` is reached.
    """
    parent: dict[Any, Any] = {}

    def find(x: Any) -> Any:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # union on the raw image id, NOT namespaced by role: in Shoes the same image is a
    # reference in one pair and a target in another, and role-namespaced nodes would
    # never merge, letting that image land on both sides of the split.
    for q in queries:
        union(q.ref_id, q.target)

    groups: dict[Any, list[BenchQuery]] = {}
    for q in queries:
        groups.setdefault(find(q.ref_id), []).append(q)

    comps = list(groups.values())
    random.Random(seed).shuffle(comps)

    want = int(round(val_frac * len(queries)))
    val: list[BenchQuery] = []
    train: list[BenchQuery] = []
    for comp in comps:
        (val if len(val) < want else train).extend(comp)
    return train, val


def _no_carve(split: str, queries: list[BenchQuery], id_to_path: dict[Any, str],
              name: str) -> Benchmark:
    """val_frac <= 0: the whole train split is kept (literature protocol, no held-out val)."""
    if split == "val":
        raise ValueError(
            f"{name}: val_frac<=0 leaves no validation split. The published protocol for this "
            "dataset selects on the test split instead -- pass --protocol literature to the "
            "trainer, which validates on test explicitly.")
    index_ids = sorted(id_to_path, key=str)
    return Benchmark(name, index_ids, [id_to_path[i] for i in index_ids], queries)


def _images_of(queries: Sequence[BenchQuery], id_to_path: dict[Any, str]) -> tuple[list, list[str]]:
    """gallery restricted to the images this query set touches (used for carved splits)."""
    ids = sorted({q.target for q in queries} | {q.ref_id for q in queries}, key=str)
    return ids, [id_to_path[i] for i in ids]


def build_shoes(data_root: str | Path, split: str = "test", seed: int = SPLIT_SEED,
                val_frac: float = VAL_FRACTION) -> Benchmark:
    """Shoes (Guo et al.) relative-caption pairs over the Attribute Discovery images.

    Images live in nested ``images/womens_*/<n>/`` directories, so they are indexed by
    basename. ``train_im_names.txt`` / ``eval_im_names.txt`` define the official
    train/test image split; ``test`` here is the official eval split.
    """
    root = Path(data_root) / "shoes"
    by_name = {p.name: str(p) for p in (root / "images").rglob("*.jpg")}
    split_file = root / ("eval_im_names.txt" if split == "test" else "train_im_names.txt")
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    pool = set(names)

    pairs = json.loads((root / "relative_captions_shoes.json").read_text())
    queries = [
        BenchQuery(qid=str(i), ref_path=by_name[p["ReferenceImageName"]],
                   caption=p["RelativeCaption"], target=p["ImageName"],
                   ref_id=p["ReferenceImageName"])
        for i, p in enumerate(pairs)
        if p["ImageName"] in pool and p["ReferenceImageName"] in pool
    ]

    if split == "test":
        index_ids = sorted(pool)
        return Benchmark("shoes", index_ids, [by_name[n] for n in index_ids], queries)
    if val_frac <= 0:
        return _no_carve(split, queries, {n: by_name[n] for n in pool}, "shoes")

    train_q, val_q = carve_val(queries, seed=seed, val_frac=val_frac)
    keep = val_q if split == "val" else train_q
    index_ids, index_paths = _images_of(keep, by_name)
    return Benchmark("shoes", index_ids, index_paths, keep)
