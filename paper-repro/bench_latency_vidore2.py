import argparse
import json
import os
import time
import statistics as stats
from collections import defaultdict
from functools import partial
from types import MethodType

import torch
from tqdm import tqdm

from dataset_configs import Datasets, RagDataset, DataSplit
from embedding_configs import all_embedders, Embedders, Modality, EncoderConfig
from fusion_methods import (
    reciprocal_rank_fusion,
    average_ranking_fusion,
    sim_score_fusion,
    normalize_min_max,
    normalize_softmax,
)
from retriever import Retriever
from utils import get_device, set_seed
from query_optimizations import OptimizationFunctions, kl_divergence, all_optimization_funcs


def sync_accel():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _is_text(r):
    return r.modality == Modality.TEXT


def _is_vision(r):
    return r.modality == Modality.VISION


def draw_random_query(dataset: RagDataset):
    idx = torch.randperm(dataset.num_queries)
    print(idx[:1])
    dataset.get_queries_indices = MethodType(lambda self, split: idx[:1], dataset)
    query = dataset.benchmark['question'][dataset.get_queries_indices(None).numpy()].tolist()
    return query


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", choices=all_embedders,
                    default=["nvidia", "colnomic", "jina_multi", "linq", "qwen_text", "jina_text_multi"])
    ap.add_argument("--values", type=int, default=10)
    ap.add_argument("--opt_func", nargs="+", choices=all_optimization_funcs,
                    default=[OptimizationFunctions.union_no_search.name])
    ap.add_argument("--opt_steps", nargs="+", default=[25], type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--datasets_path_prefix', default='')
    ap.add_argument("--out_dir", default="latency_results_vidore2")
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    vidore2 = [
        Datasets.esg_reports_v2,
        Datasets.biomedical_lectures_v2,
        Datasets.economics_reports_v2,
        Datasets.esg_reports_human_labeled_v2,
    ]

    embedder_cfgs: list[EncoderConfig] = [getattr(Embedders, m) for m in args.embedders]
    retrievers = [Retriever(cfg) for cfg in embedder_cfgs]
    out_dir = os.path.join("output", "latency", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    runner = SimpleRunner()

    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "mps": torch.backends.mps.is_available(),
        "embedders": " ".join([ret.id for ret in retrievers]),
        "values": args.values,
    }

    for ds in tqdm(vidore2):
        dataset = RagDataset(ds, prefix=args.datasets_path_prefix)

        for i in range(len(retrievers) - 1):
            mod1 = retrievers[i]
            if not _is_vision(mod1):
                continue  # mod1 must be vision

            embedder_cfg1 = embedder_cfgs[i]
            embedder1 = embedder_cfg1.embedder()

            # bench embedding of queries
            embedder1.load_model(model_id=embedder_cfg1.model_id)
            for _ in range(args.values):
                sampled_query = draw_random_query(dataset)
                runner.bench(
                    f"calc_embeddings[{dataset.id}][{mod1.id}]",
                    partial(embedder1.calc_query_embeddings, queries=sampled_query),
                    group=f"{mod1.id}_embed",
                    dataset_id=dataset.id,
                )
            del embedder1.model
            if hasattr(embedder1, 'processor'):  # nomic
                del embedder1.processor

            for j in range(i + 1, len(retrievers)):
                mod2 = retrievers[j]
                if not _is_text(mod2):
                    continue  # mod2 must be text

                embedder_cfg2 = embedder_cfgs[j]
                embedder2 = embedder_cfg2.embedder()

                # bench embedding of queries
                embedder2.load_model(model_id=embedder_cfg2.model_id)
                for _ in range(args.values):
                    sampled_query = draw_random_query(dataset)
                    runner.bench(
                        f"calc_embeddings[{dataset.id}][{mod2.id}]",
                        partial(embedder2.calc_query_embeddings, queries=sampled_query),
                        group=f"{mod2.id}_embed",
                        dataset_id=dataset.id,
                    )
                del embedder2.model

                # bench retrieval and hybrids
                for _ in range(args.values):
                    draw_random_query(dataset)
                    mod1.load_embs(dataset)
                    mod2.load_embs(dataset)

                    # Benchmark & run retrievals
                    runner.bench(
                        f"run_retrieval[{dataset.id}][{mod1.id}]",
                        partial(mod1.run_retrieval, dataset),
                        group=f"{mod1.id}_retrieve",
                        dataset_id=dataset.id,
                    )
                    r1, _ = mod1.run_retrieval(dataset)

                    runner.bench(
                        f"run_retrieval[{dataset.id}][{mod2.id}]",
                        partial(mod2.run_retrieval, dataset),
                        group=f"{mod2.id}_retrieve",
                        dataset_id=dataset.id,
                    )
                    r2, _ = mod2.run_retrieval(dataset)

                    # Fusion baselines for this (vision, text) pair
                    runner.bench(
                        f"fusion[RRF][{dataset.id}][{mod1.id}+{mod2.id}]",
                        partial(reciprocal_rank_fusion, r1=r1, r2=r2),
                        group=f"RRF-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[average][{dataset.id}][{mod1.id}+{mod2.id}]",
                        partial(average_ranking_fusion, r1=r1, r2=r2),
                        group=f"average-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[sim_score_minmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                        partial(sim_score_fusion, r1=r1, r2=r2, normal_func=normalize_min_max),
                        group=f"sim_score_minmax-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )
                    runner.bench(
                        f"fusion[sim_score_softmax][{dataset.id}][{mod1.id}+{mod2.id}]",
                        partial(sim_score_fusion, r1=r1, r2=r2, normal_func=normalize_softmax),
                        group=f"sim_score_softmax-{mod1.id}-{mod2.id}",
                        dataset_id=dataset.id,
                    )

                    for opt_func_name in args.opt_func:
                        opt_func = OptimizationFunctions[opt_func_name].value
                        for n_steps in args.opt_steps:
                            runner.bench(
                                f"query_opt[{opt_func_name} n={n_steps}][{dataset.id}][{mod1.id}<-{mod2.id}]",
                                partial(
                                    opt_func,
                                    main_model=mod1, feedback_model=mod2, dataset=dataset,
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
                                group=f"{mod1.id}-feedback-from-{mod2.id}",
                                dataset_id=dataset.id,
                            )

                    mod1.clean_embs(dataset)
                    mod2.clean_embs(dataset)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    sync_accel()

    if runner.results:
        runner.aggregate(replace=True)
        out_json = os.path.join(out_dir, "suite.json")
        runner.dump(out_json, extra_meta=meta)
    else:
        print(f"No latency results calculated (embedders: {args.embedders})")


if __name__ == "__main__":
    main()
