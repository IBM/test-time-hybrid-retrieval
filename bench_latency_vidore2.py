import argparse
import json
import os
import time
import statistics as stats
from functools import partial

import os, time, json, statistics as stats, argparse
import torch
from types import MethodType

from dataset_configs import Datasets, RagDataset, DataSplit
from embedding_configs import Embedders, Modality
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
from collections import defaultdict
from tqdm import tqdm

def sync_accel():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _is_text(r):
    return (r.modality == Modality.TEXT)


def _is_vision(r):
    return (r.modality == Modality.VISION)


class SimpleRunner:
    def __init__(self):
        self.results = []

    def bench(self, name, fn, *, inner_loops=1, group=None, dataset_id=None):
        # Warmup:
        fn()
        sync_accel()
        # Measure
        t0 = time.perf_counter()
        fn()
        sync_accel()
        dt = (time.perf_counter() - t0)*1000
        r = {
            "name": name,
            "group": group,
            "dataset": dataset_id,
            "values": [dt],
            "mean": dt,
        }
        self.results.append(r)
        return r

    def aggregate(self, replace=False):
        by_group = defaultdict(list)
        for r in self.results:
            g = r.get("group")
            if g:
                by_group[g].extend(r["values"])
        aggs = []
        for g, vals in by_group.items():
            aggs.append({
                "name": g,
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
    ap.add_argument("--embedders", nargs="+", default=["colnomic", "jina_multi","linq","qwen_text","jina_text_multi"])
    ap.add_argument("--values", type=int, default=10)
    ap.add_argument("--opt_func", nargs="+", default=["union"])
    ap.add_argument("--opt_steps", nargs="+", default=[25], type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="latency_results_vidore2")
    args = ap.parse_args()

    FUNC_MAP = {
        "union": OptimizationFunctions["union_no_search"].value,
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
    runner = SimpleRunner()

    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "mps": torch.backends.mps.is_available(),
        "embedders": " ".join(args.embedders),
        "values": args.values,
    }

    for ds in tqdm(vidore2):
        dataset = RagDataset(ds, prefix=prefix)

        def draw_random_query(dataset):
            idx = torch.randperm(dataset.num_queries)
            print(idx[:1])
            dataset.get_queries_indices = MethodType(lambda self,split: idx[:1], dataset)

        for i in range(len(retrievers) - 1):
            mod1 = retrievers[i]
            if not _is_vision(mod1):
                continue  # mod1 must be vision

            for j in range(i + 1, len(retrievers)):
                mod2 = retrievers[j]
                if not _is_text(mod2):
                    continue  # mod2 must be text

                for _ in range(args.values):
                    draw_random_query(dataset)
                    mod1.load_embs(dataset)
                    mod2.load_embs(dataset)

                    # Benchmark & run retrievals
                    runner.bench(
                        f"run_retrieval[{dataset.id}][{mod1.id}]",
                        lambda m=mod1, d=dataset: m.run_retrieval(d),
                        group=f"{mod1.id}",
                        dataset_id=dataset.id,
                    )
                    r1, _ = mod1.run_retrieval(dataset)

                    runner.bench(
                        f"run_retrieval[{dataset.id}][{mod2.id}]",
                        lambda m=mod2, d=dataset: m.run_retrieval(d),
                        group=f"{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    r2, _ = mod2.run_retrieval(dataset)

                    # Fusion baselines for this (vision, text) pair
                    runner.bench(
                        f"fusion[RRF][{dataset.id}][{mod1.id}+{mod2.id}]",
                        lambda a=r1, b=r2: reciprocal_rank_fusion(a, b),
                        group=f"RRF-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[average][{dataset.id}][{mod1.id}+{mod2.id}]",
                        lambda a=r1, b=r2: average_ranking_fusion(a, b),
                        group=f"average-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[sim_score_minmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                    lambda a=r1, b=r2: sim_score_fusion(a, b, normal_func=normalize_min_max),
                        group=f"sim_score_minmax-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[sim_score_softmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                        lambda a=r1, b=r2: sim_score_fusion(a, b, normal_func=normalize_softmax),
                        group=f"sim_score_softmax-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )

                    for opt_func in args.opt_func:
                        for n_steps in args.opt_steps:
                            runner.bench(
                                f"query_opt[{opt_func} n={n_steps}][{dataset.id}][{mod1.id}<-{mod2.id}]",
                                lambda m1=mod1, m2=mod2, d=dataset, of=opt_func, ns=n_steps: FUNC_MAP[of](
                                    m1, m2, d,
                                    device=device,
                                    k=10,
                                    lr=0.001,
                                    n_steps=ns,
                                    T=1,
                                    loss_func=kl_divergence,
                                    mixture_alpha=0.5,
                                    optimizer=torch.optim.Adam,
                                    split=DataSplit.TEST,
                                ),
                                group=f"{mod1.id}-feedback-from-{mod2.id}",
                                dataset_id=dataset.id,
                            )

                    mod1.clean_embs(dataset)
                    mod2.clean_embs(dataset)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    sync_accel()

    runner.aggregate(replace=True)
    out_json = os.path.join(args.out_dir, "suite.json")
    runner.dump(out_json, extra_meta=meta)


if __name__ == "__main__":
    main()


