import argparse
import json
import os
import time
import statistics as stats
from functools import partial

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
from query_optimizations import OptimizationFunctions,  kl_divergence


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

    def bench(self, name, fn, *, inner_loops=1):
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
            "values": samples,
            "mean": stats.fmean(samples) if samples else None,
            "median": stats.median(samples) if samples else None,
            "min": min(samples) if samples else None,
            "stdev": stats.pstdev(samples) if len(samples) > 1 else 0.0,
        }
        self.results.append(r)
        return r

    def dump(self, path, extra_meta=None):
        out = {"benchmarks": self.results, "metadata": extra_meta or {}}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", choices=all_embedders,
                    default=[Embedders.colnomic, Embedders.linq, Embedders.qwen_text])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--values", type=int, default=3)
    ap.add_argument("--opt_func", nargs="+", choices=all_optimization_funcs,
                    default=[OptimizationFunctions.union_with_search.name])
    ap.add_argument("--opt_steps", nargs="+", default=[10, 25, 50])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="latency_results_vidore2")
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    prefix = "/proj/omri/" if on_ccc() else ""
    dataset = RagDataset(Datasets.economics_reports_v2, prefix=prefix)

    retrievers = [Retriever(m) for m in args.embedders]
    for r in retrievers:
        r.load_embs(dataset)

    os.makedirs(args.out_dir, exist_ok=True)
    runner = SimpleRunner(values=args.values, warmups=args.warmups)

    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "mps": torch.backends.mps.is_available(),
        "dataset": dataset.id,
        "embedders": " ".join([ret.id for ret in retrievers]),
        "values": args.values,
        "warmups": args.warmups,
    }

    results = []

    # per-model retrieval timing
    for i in range(len(retrievers) - 1):
        mod1 = retrievers[i]
        res = runner.bench(
            f"run_retrieval[{dataset.id}][{mod1.id}]",
            partial(mod1.run_retrieval, dataset)
        )
        results.append(res)
        r1, top_idx_1 = mod1.run_retrieval(dataset)

        for j in range(i + 1, len(retrievers)):
            mod2 = retrievers[j]

            res = runner.bench(
                f"run_retrieval[{dataset.id}][{mod2.id}]",
                partial(mod2.run_retrieval, dataset)
            )
            results.append(res)
            r2, top_idx_2 = mod2.run_retrieval(dataset)

            if not mod1.is_sparse:
                runner.bench(
                    f"fusion[scores_feedback][{dataset.id}][{mod1.id}<-{mod2.id}]",
                    partial(scores_feedback,
                            main_model=mod1, feedback_model=mod2, dataset=dataset, top_k_idxs=top_idx_1)
                )
            if not mod2.is_sparse:
                runner.bench(
                    f"fusion[scores_feedback][{dataset.id}][{mod2.id}<-{mod1.id}]",
                    partial(scores_feedback,
                            main_model=mod2, feedback_model=mod1, dataset=dataset, top_k_idxs=top_idx_2)
                )

            runner.bench(
                f"fusion[RRF][{dataset.id}][{mod1.id}+{mod2.id}]",
                partial(reciprocal_rank_fusion, r1=r1, r2=r2)
            )
            runner.bench(
                f"fusion[average][{dataset.id}][{mod1.id}+{mod2.id}]",
                partial(average_ranking_fusion, r1=r1, r2=r2)
            )

            runner.bench(
                f"fusion[sim_score_minmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                partial(sim_score_fusion, r1=r1, r2=r2, normal_func=normalize_min_max)
            )
            runner.bench(
                f"fusion[sim_score_softmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                partial(sim_score_fusion, r1=r1, r2=r2, normal_func=normalize_softmax)
            )
            for opt_func in args.opt_func:
                for n_steps in args.opt_steps:
                    if not mod1.is_sparse:
                        runner.bench(
                            f"query_opt[{opt_func} n={n_steps}]"
                            f"[{dataset.id}][{mod1.id}<-{mod2.id}]",
                            partial(OptimizationFunctions[opt_func].value,
                                    main_model=mod1, feedback_model=mod2, dataset=dataset,
                                    device=device,
                                    k=10,
                                    lr=0.001,
                                    n_steps=n_steps,
                                    T=1,
                                    mixture_alpha=0.5,
                                    loss_func=kl_divergence,
                                    optimizer=torch.optim.Adam,
                                    split=DataSplit.TEST
                            )
                        )

                    if not mod2.is_sparse:
                        runner.bench(
                            f"query_opt[{opt_func} n={n_steps}]"
                            f"[{dataset.id}][{mod1.id}<-{mod2.id}]",
                            partial(OptimizationFunctions[opt_func].value,
                                    main_model=mod2, feedback_model=mod1, dataset=dataset,
                                    device=device,
                                    k=10,
                                    lr=0.001,
                                    n_steps=n_steps,
                                    T=1,
                                    mixture_alpha=0.5,
                                    loss_func=kl_divergence,
                                    optimizer=torch.optim.Adam,
                                    split=DataSplit.TEST
                            )
                        )

    for r in retrievers:
        r.clean_embs(dataset)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    sync_accel()

    out_json = os.path.join(args.out_dir, "suite.json")
    runner.dump(out_json, extra_meta=meta)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
