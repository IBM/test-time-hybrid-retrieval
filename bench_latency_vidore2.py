import argparse
import json
import os
import time
import statistics as stats
from functools import partial

import os, time, json, statistics as stats, argparse
import torch

from dataset_configs import Datasets, RagDataset, DataSplit
from embedding_configs import Embedders, all_embedders
from fusion_methods import (
    reciprocal_rank_fusion,
    average_ranking_fusion,
    sim_score_fusion,
    normalize_min_max,
    normalize_softmax,
)
from query_optimizations import scores_feedback, all_optimization_funcs
from retriever import Retriever
from utils import get_device, set_seed, on_ccc
from query_optimizations import OptimizationFunctions, kl_divergence


def sync_accel():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


class SimpleRunner:
    def __init__(self, values=5, warmups=1):
        self.values = values
        self.warmups = warmups
        self.results = []

    def bench(self, name, fn, *, inner_loops=1, group=None, dataset_id=None):
        for _ in range(self.warmups):
            fn()
            sync_accel()
        samples = []
        for _ in range(self.values):
            t0 = time.perf_counter()
            fn()
            sync_accel()
            dt = time.perf_counter() - t0
            samples.append(dt / inner_loops)
        r = {
            "name": name,
            "group": group,
            "dataset": dataset_id,
            "values": samples,
            "mean": stats.fmean(samples) if samples else None,
        }
        self.results.append(r)
        return r

    def aggregate(self, replace=False):
        from collections import defaultdict
        by_group = defaultdict(list)
        for r in self.results:
            g = r.get("group")
            if g:
                by_group[g].extend(r["values"])
        aggs = []
        for g, vals in by_group.items():
            aggs.append({
                "name": f"AGG::{g}",
                "values": vals,
                "mean": stats.fmean(vals) if vals else None,
                "median": stats.median(vals) if vals else None,
                "min": min(vals) if vals else None,
                "stdev": stats.pstdev(vals) if len(vals) > 1 else 0.0,
            })
        if replace:
            self.results = aggs
        else:
            self.results.extend(aggs)

    def dump(self, path, extra_meta=None):
        out = {"benchmarks": self.results, "metadata": extra_meta or {}}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


def parse_embedders(names):
    out = []
    for n in names:
        e = getattr(Embedders, n, None)
        if e is None:
            raise ValueError(f"Unknown embedder: {n}")
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", default=["colnomic", "linq"])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--values", type=int, default=3)
    ap.add_argument("--opt_func", nargs="+", default=["union"])
    ap.add_argument("--opt_steps", nargs="+", default=[10], type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="latency_results_vidore2")
    args = ap.parse_args()

    FUNC_MAP = {
        "union": OptimizationFunctions.optimize_queries_union,
        "main": OptimizationFunctions.optimize_queries_main,
    }

    device = get_device()
    set_seed(args.seed)
    prefix = "/proj/omri/" if on_ccc() else ""
    vidore2 = [
        Datasets.esg_reports_v2,
        Datasets.biomedical_lectures_v2,
        Datasets.economics_reports_v2,
        Datasets.esg_reports_human_labeled_v2,
    ]

    retrievers = [Retriever(m) for m in parse_embedders(args.embedders)]
    os.makedirs(args.out_dir, exist_ok=True)
    runner = SimpleRunner(values=args.values, warmups=args.warmups)

    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "mps": torch.backends.mps.is_available(),
        "embedders": " ".join(args.embedders),
        "values": args.values,
        "warmups": args.warmups,
    }

    for ds in vidore2:
        dataset = RagDataset(ds, prefix=prefix)
        for r in retrievers:
            r.load_embs(dataset)

        for i in range(len(retrievers) - 1):
            mod1 = retrievers[i]
            runner.bench(
                f"run_retrieval[{dataset.id}][{mod1.id}]",
                lambda m=mod1, d=dataset: m.run_retrieval(d),
                group=f"run_retrieval[{mod1.id}]",
                dataset_id=dataset.id,
            )
            r1, top_idx_1 = mod1.run_retrieval(dataset)

            for j in range(i + 1, len(retrievers)):
                mod2 = retrievers[j]

                runner.bench(
                    f"run_retrieval[{dataset.id}][{mod2.id}]",
                    lambda m=mod2, d=dataset: m.run_retrieval(d),
                    group=f"run_retrieval[{mod2.id}]",
                    dataset_id=dataset.id,
                )
                r2, top_idx_2 = mod2.run_retrieval(dataset)

                if not mod1.is_sparse:
                    runner.bench(
                        f"fusion[scores_feedback][{dataset.id}][{mod1.id}<-{mod2.id}]",
                        lambda m1=mod1, m2=mod2, dset=dataset, ti=top_idx_1: scores_feedback(m1, m2, dset, ti),
                        group=f"fusion[scores_feedback][{mod1.id}<-{mod2.id}]",
                        dataset_id=dataset.id,
                    )
                if not mod2.is_sparse:
                    runner.bench(
                        f"fusion[scores_feedback][{dataset.id}][{mod2.id}<-{mod1.id}]",
                        lambda m1=mod2, m2=mod1, dset=dataset, ti=top_idx_2: scores_feedback(m1, m2, dset, ti),
                        group=f"fusion[scores_feedback][{mod2.id}<-{mod1.id}]",
                        dataset_id=dataset.id,
                    )

                runner.bench(
                    f"fusion[RRF][{dataset.id}][{mod1.id}+{mod2.id}]",
                    lambda r1=r1, r2=r2: reciprocal_rank_fusion(r1, r2),
                    group=f"fusion[RRF][{mod1.id}+{mod2.id}]",
                    dataset_id=dataset.id,
                )
                runner.bench(
                    f"fusion[average][{dataset.id}][{mod1.id}+{mod2.id}]",
                    lambda r1=r1, r2=r2: average_ranking_fusion(r1, r2),
                    group=f"fusion[average][{mod1.id}+{mod2.id}]",
                    dataset_id=dataset.id,
                )
                runner.bench(
                    f"fusion[sim_score_minmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                    lambda r1=r1, r2=r2: sim_score_fusion(r1, r2, normal_func=normalize_min_max),
                    group=f"fusion[sim_score_minmax][{mod1.id}+{mod2.id}]",
                    dataset_id=dataset.id,
                )
                runner.bench(
                    f"fusion[sim_score_softmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                    lambda r1=r1, r2=r2: sim_score_fusion(r1, r2, normal_func=normalize_softmax),
                    group=f"fusion[sim_score_softmax][{mod1.id}+{mod2.id}]",
                    dataset_id=dataset.id,
                )

                for opt_func in args.opt_func:
                    for n_steps in args.opt_steps:
                        if not mod1.is_sparse:
                            runner.bench(
                                f"query_opt[{opt_func} n={n_steps}][{dataset.id}][{mod1.id}<-{mod2.id}]",
                                lambda m1=mod1, m2=mod2, d=dataset: FUNC_MAP[opt_func](
                                    m1, m2, d,
                                    device=device,
                                    k=10,
                                    lr=0.001,
                                    n_steps=n_steps,
                                    T=1,
                                    loss_func=kl_divergence,
                                    mixture_alpha=0.5,
                                    optimizer=torch.optim.Adam,
                                    split=DataSplit.TEST,
                                ),
                                group=f"query_opt[{opt_func} n={n_steps}][{mod1.id}<-{mod2.id}]",
                                dataset_id=dataset.id,
                            )
                        if not mod2.is_sparse:
                            runner.bench(
                                f"query_opt[{opt_func} n={n_steps}][{dataset.id}][{mod2.id}<-{mod1.id}]",
                                lambda m1=mod2, m2=mod1, d=dataset: FUNC_MAP[opt_func](
                                    m1, m2, d,
                                    device=device,
                                    k=10,
                                    lr=0.001,
                                    n_steps=n_steps,
                                    T=1,
                                    loss_func=kl_divergence,
                                    mixture_alpha=0.5,
                                    optimizer=torch.optim.Adam,
                                    split=DataSplit.TEST,
                                ),
                                group=f"query_opt[{opt_func} n={n_steps}][{mod2.id}<-{mod1.id}]",
                                dataset_id=dataset.id,
                            )

        for r in retrievers:
            r.clean_embs(dataset)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sync_accel()

    runner.aggregate(replace=True)
    out_json = os.path.join(args.out_dir, "suite.json")
    runner.dump(out_json, extra_meta=meta)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
