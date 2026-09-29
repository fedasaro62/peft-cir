"""Score a retriever on CIRR, CIRCO, FashionIQ or Shoes.

    python tools/evaluate.py --arch mti --benchmark cirr --split val
    python tools/evaluate.py --benchmark all --peft_checkpoint outputs/CIRR/mti_peft/lora_text_vision

Without ``--peft_checkpoint`` this evaluates the frozen (zero-shot) model. ``val`` computes
metrics from local ground truth and appends a row to ``<output>/benchmarks.csv``; ``test``
writes eval-server submission rankings for cirr/circo/fiq, whose test GT is hidden, but
computes metrics for shoes, whose GT is local.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from peft_cir import ROOT
from peft_cir.data import metrics
from peft_cir.data.benchmarks import (
    build_circo,
    build_cirr,
    build_fiq,
    build_shoes,
    write_cirr_submission,
    write_ranked_submission,
)
from peft_cir.data.datasets import FIQ_SUBTASKS
from peft_cir.evaluation import rank_benchmark
from peft_cir.registry import ARCHS, build_model
from peft_cir.training import loss_tag_suffix

BENCHMARKS = ["cirr", "circo", "fiq", "shoes"]
# benchmarks whose held-out split ships local ground truth, so `test` is scored here rather
# than written out as a submission
LOCAL_GT = {"shoes"}
METRIC_FN = {"cirr": metrics.cirr_metrics, "circo": metrics.circo_map,
             "shoes": metrics.shoes_recall}


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="all", choices=["all"] + BENCHMARKS,
                   help="'all' runs cirr, circo and fiq.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--data_root", default=str(ROOT / "data"))
    p.add_argument("--clip_model_name", default="large", choices=["base", "large", "huge", "giga"])
    p.add_argument("--arch", default="searle", choices=ARCHS,
                   help="which retriever to evaluate; ignored when --peft_checkpoint records one.")
    p.add_argument("--phi_checkpoint", default=None,
                   help="mapping-network checkpoint; defaults to the one --arch ships with.")
    p.add_argument("--cache_dir", default=str(ROOT / "resources" / "pretrained" / "hf_models"))
    p.add_argument("--model_tag", default="searle")
    p.add_argument("--peft_checkpoint", default=None,
                   help="a tools/train.py run dir (best.pt + meta.json); evaluates the "
                        "fine-tuned model instead of the frozen one.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output", default=str(ROOT / "outputs" / "CIRR" / "searle"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(args):
    """Returns ``(model, preprocess, model_tag, method_tag, run_meta)``."""
    if not args.peft_checkpoint:
        model, preprocess = build_model(args.arch, args.clip_model_name, args.phi_checkpoint,
                                        "frozen", "text", cache_dir=args.cache_dir)
        model.to(args.device).eval()
        return model, preprocess, f"{args.model_tag}_{args.clip_model_name}", "frozen", {}

    meta = json.loads((Path(args.peft_checkpoint) / "meta.json").read_text())
    # runs predating --arch carry no "arch" field and are all SEARLE/LinCIR; runs predating
    # --loss carry no "loss" field and were all trained symmetric
    model, preprocess = build_model(
        meta.get("arch", "searle"), meta["clip_model_name"], meta["phi_checkpoint"],
        meta["method"], meta["target"], cache_dir=args.cache_dir, rank=meta["rank"],
        alpha=meta["alpha"], dropout=meta["dropout"], adapter_dim=meta["adapter_dim"],
        num_virtual_tokens=meta["num_virtual_tokens"])
    model.load_trainable(torch.load(Path(args.peft_checkpoint) / "best.pt", map_location="cpu"))
    model.to(args.device).eval()
    method = f'{meta["method"]}_{meta["target"]}{loss_tag_suffix(meta.get("loss", "symmetric"))}'
    return model, preprocess, meta.get("model_tag", "searle"), method, meta


def build_benchmark(name: str, data_root: Path, split: str):
    """The benchmark object for ``name``."""
    return {"cirr": build_cirr, "circo": build_circo, "shoes": build_shoes}[name](data_root, split)


def score(model, preprocess, bench, device, bs) -> dict:
    ranked, subset = rank_benchmark(model, preprocess, bench, device, bs, 50)
    if bench.name == "cirr":
        return metrics.cirr_metrics(ranked, subset, bench.queries)
    return METRIC_FN.get(bench.name, metrics.fiq_recall)(ranked, bench.queries)


def append_row(output_dir: str, row: dict) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    csv = path / "benchmarks.csv"
    df = pd.DataFrame([row])
    if csv.exists():
        df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
    df.to_csv(csv, index=False)


def main() -> None:
    args = parse_args()
    model, preprocess, tag, method, run_meta = load_model(args)
    data_root = Path(args.data_root)
    test = args.split == "test"
    sub_dir = Path(args.output) / "submissions" / f"{tag}__{method}"
    row = {"model": tag, "method": method, "split": args.split}

    def report(name: str, bench) -> None:
        m = score(model, preprocess, bench, args.device, args.batch_size)
        extra = ({"protocol": run_meta.get("protocol") or "frozen",
                  "selected_on": run_meta.get("selected_on") or "-",
                  "queries": len(bench.queries), "index": len(bench.index_ids)}
                 if name in LOCAL_GT else {})
        append_row(args.output, {**row, "benchmark": name, **extra, **m})
        print(f"{name} ({args.split}):", m)

    for name in (["cirr", "circo", "fiq"] if args.benchmark == "all" else [args.benchmark]):
        if name == "fiq":
            per_sub = {}
            for subt in FIQ_SUBTASKS:
                bench = build_fiq(data_root, subt, args.split)
                if test:
                    ranked, _ = rank_benchmark(model, preprocess, bench, args.device,
                                               args.batch_size, 50)
                    write_ranked_submission(sub_dir, f"fashioniq-{subt}", ranked, bench.queries, 50)
                    print(f"fashioniq-{subt} test rankings -> {sub_dir}")
                else:
                    per_sub[subt] = score(model, preprocess, bench, args.device, args.batch_size)
                    append_row(args.output, {**row, "benchmark": f"fashioniq-{subt}",
                                             **per_sub[subt]})
                    print(f"fashioniq-{subt}:", per_sub[subt])
            if per_sub:
                avg = {k: sum(s[k] for s in per_sub.values()) / len(per_sub)
                       for k in per_sub["dress"]}
                append_row(args.output, {**row, "benchmark": "fashioniq-avg", **avg})
                print("fashioniq-avg:", avg)
            continue

        bench = build_benchmark(name, data_root, args.split)
        if not test or name in LOCAL_GT:
            report(name, bench)
        elif name == "cirr":
            ranked, subset = rank_benchmark(model, preprocess, bench, args.device,
                                            args.batch_size, 50)
            write_cirr_submission(sub_dir, ranked, subset, bench.queries)
            print(f"cirr test rankings -> {sub_dir}")
        else:
            ranked, _ = rank_benchmark(model, preprocess, bench, args.device, args.batch_size, 50)
            write_ranked_submission(sub_dir, name, ranked, bench.queries, 50)
            print(f"{name} test rankings -> {sub_dir}")


if __name__ == "__main__":
    main()
