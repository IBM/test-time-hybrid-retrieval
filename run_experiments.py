import argparse
import ast
import gc
import json
import os
from collections import defaultdict
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from typing import Callable

import numpy as np
import pandas as pd
import torch
import tqdm

from dataset_configs import Datasets, DataSplit, RagDataset
from embedding_configs import Embedders
from fusion_methods import average_ranking_fusion, normalize_softmax, normalize_min_max, reciprocal_rank_fusion, \
    sim_score_fusion
from query_optimizations import OptimizationFunctions, scores_feedback, kl_divergence, js_divergence
from retriever import Retriever
from utils import get_device, get_run_hash, set_seed, on_ccc


def oracle_retriever(dataset: RagDataset):
    b = dataset.benchmark
    oracle_res = {q: [(k, v) for k, v in r.items()]
                  for q, r in zip(b['question'], b['correct_answer_document_ids'])}
    return oracle_res


def run_baselines(dataset: RagDataset, mod_1, mod_2, r1, r2, top_idx_1, top_idx_2, target_metric='ndcg@5'):
    results = []
    info_dict = {
        "lr": 0.0, "k": 0, "n_steps": 0, "temp": 0.0,
        "mixture_alpha": 0, "loss_func": "baseline",
        "weight": 0,
        "optimization_func": "", "optimizer": "",
        "dataset": dataset.id,
    }

    results.append({
        **info_dict,
        "run_id": mod_1.id,
        "main_model": mod_1.id,
        "feedback_model": "",
        "metrics": dataset.evaluate(r1)
    })

    results.append({
        **info_dict,
        "run_id": mod_2.id,
        "main_model": mod_2.id,
        "feedback_model": "",
        "metrics": dataset.evaluate(r2)
    })

    oracle_r = oracle_retriever(dataset)
    results.append({
        **info_dict,
        "run_id": "oracle",
        "main_model": "",
        "feedback_model": "",
        "metrics": dataset.evaluate(oracle_r)
    })

    late_pipelines = {
        "RRF": reciprocal_rank_fusion,
        "average": average_ranking_fusion,
    }

    tunable_pipelines = {
        "sim_score_minmax": partial(sim_score_fusion, normal_func=normalize_min_max),
        "sim_score_softmax": partial(sim_score_fusion, normal_func=normalize_softmax),
        f"scores_feedback_{mod_1.id}-{mod_2.id}": partial(scores_feedback, main_model=mod_1, feedback_model=mod_2,
                                                          dataset=dataset, top_k_idxs=top_idx_1),
        f"scores_feedback_{mod_2.id}-{mod_1.id}": partial(scores_feedback, main_model=mod_2, feedback_model=mod_1,
                                                          dataset=dataset, top_k_idxs=top_idx_2),
        "weighted_average": average_ranking_fusion,
        "weighted_RRF": reciprocal_rank_fusion,
    }

    if args.tune:
        r1_dev, _ = mod_1.run_retrieval(dataset, split=DataSplit.DEV)
        r2_dev, _ = mod_2.run_retrieval(dataset, split=DataSplit.DEV)

        for pipeline, fusion_func in tunable_pipelines.items():
            best_val, best_alpha = 0, 0
            for alpha in [0.1*i for i in range(1, 10)]:
                if "scores_feedback" in pipeline:
                    r = fusion_func(alpha=alpha, split=DataSplit.DEV)
                else:
                    r = fusion_func(r1_dev, r2_dev, alpha=alpha)

                eval = dataset.evaluate(r)
                if eval[target_metric] > best_val:
                    best_val = eval[target_metric]
                    best_alpha = alpha

            if "scores_feedback" in pipeline:
                r = fusion_func(alpha=best_alpha)
                run_id = pipeline
            else:
                r = fusion_func(r1, r2, alpha=best_alpha)
                run_id = f"{pipeline}-{mod_1.id}-{mod_2.id}"

            results.append({
                **info_dict,
                "run_id": run_id,
                "weight": best_alpha,
                "main_model": "",
                "feedback_model": "",
                "metrics": dataset.evaluate(r)
            })
    else:
        for pipeline, fusion_func in tunable_pipelines.items():
            for alpha in [0.1*i for i in range(1, 10)]:
                late_pipelines[f"{pipeline}_{alpha:.2f}-{1-alpha:.2f}"] = partial(fusion_func, alpha=alpha)

    for pipeline, fusion_func in late_pipelines.items():
        if "scores_feedback" in pipeline:
            r = fusion_func()
        else:
            r = fusion_func(r1, r2)
        results.append({
            **info_dict,
            "run_id": f"{pipeline}-{mod_1.id}-{mod_2.id}",
            "main_model": "",
            "feedback_model": "",
            "metrics": dataset.evaluate(r)
        })

    return results


def tune_hyper(dataset, mod_1, mod_2, device, input_params, weight_for_feedback_model=0.5, target_metric='ndcg@5'):
    if any(x[4] == "dynamic" for x in input_params):
        input_params = [p[:4] + (weight_for_feedback_model,) + p[5:]
                        if p[4] == "dynamic" else p for p in input_params]
    dev_results = []
    for run_args in tqdm.tqdm(input_params):
        dev_results.append(run_query_optimizations(dataset, mod_1, mod_2, device, *run_args, split=DataSplit.DEV))
    max_val, best_params = 0, None
    for results, params in zip(dev_results, input_params):
        val = results['metrics'][target_metric]
        if val > max_val:
            max_val = val
            best_params = params
    print(f'best params: {best_params} the {target_metric} is {max_val}')
    return best_params


def run_query_optimizations(dataset, mod_1: Retriever, mod_2: Retriever, device, lr, k, n, t, mixture_alpha,
                            loss_func: Callable, optimizer: torch.optim.Optimizer, optimization_func_name: str,
                            split=DataSplit.TEST):
    if mod_1.is_sparse:
        result_dict = {}
    else:
        optimization_func = OptimizationFunctions[optimization_func_name].value
        r = optimization_func(
            mod_1, mod_2, dataset, device=device,
            k=k, lr=lr, n_steps=n, T=t, mixture_alpha=mixture_alpha, loss_func=loss_func,
            optimizer=optimizer, split=split)

        result_dict = {"run_id": f"{mod_1.id}-feedback-from-{mod_2.id}",
                       "main_model": mod_1.id,
                       "feedback_model": mod_2.id,
                       "metrics": dataset.evaluate(r)}

    return result_dict


def main(args):
    device = get_device()
    prefix = "/proj/omri/" if on_ccc() else ""

    # benchmarks
    vidore1 = [Datasets.arxivqa, Datasets.docvqa, Datasets.infovqa, Datasets.tabfquad, Datasets.tatdqa,
               Datasets.shiftproject, Datasets.artificial_intelligence, Datasets.energy_test,
               Datasets.government_reports, Datasets.healthcare]
    vidore2 = [
        Datasets.esg_reports_v2,
        Datasets.biomedical_lectures_v2,
        Datasets.economics_reports_v2,
        Datasets.esg_reports_human_labeled_v2
    ]
    real_mm_rag = [
        Datasets.FinReport,
        Datasets.FinSlides,
        Datasets.TechReport,
        Datasets.TechSlides
    ]

    benchmarks = {"vidore1": vidore1, "vidore2": vidore2, "real_mm_rag": real_mm_rag}

    models_in_experiment = [
        # Embedders.nvidia,
        # Embedders.jina_multi,
        # Embedders.jina_single,
        Embedders.colnomic,

        # Embedders.jina_text_single,
        # Embedders.jina_text_multi,
        Embedders.linq,
        Embedders.qwen_text,
        # Embedders.bm25,
    ]

    datasets_in_experiment = []
    for k in args.benchmarks:
        datasets_in_experiment += benchmarks[k]

    lrs = [
        5e-6,
        1e-5,
        3e-5,
        5e-5,
        1e-4,
        3e-4,
        5e-4,
        1e-3,
        3e-3,
        5e-3,
    ]
    ks = [
        10,
        # 20
        # 50,
    ]
    n_steps = [
        # 10,
        # 25,
        50,
        # 100
    ]
    Ts = [1]
    mixture = [
        "dynamic"
    ]
    loss_funcs = [
        kl_divergence,
    ]
    optimization_funcs = [
        # OptimizationFunctions.main_no_search.name,
        OptimizationFunctions.union_no_search.name,
        # OptimizationFunctions.union_sample_no_search.name,
    ]

    optimizers = [
        torch.optim.Adam,
        # torch.optim.AdamW,
        # torch.optim.Adagrad,
        # torch.optim.SGD,
        # torch.optim.RMSprop
        ]

    models_in_experiment = [Retriever(m) for m in models_in_experiment]
    h = get_run_hash(models_in_experiment, datasets_in_experiment, lrs, ks, n_steps, Ts, mixture,
                     loss_funcs, optimizers, optimization_funcs)
    out_dir = f"output/results-{h}{args.out_dir_suffix}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results in {out_dir}")

    rows = []
    for dataset_name in datasets_in_experiment:
        set_seed()
        dataset = RagDataset(dataset_name, prefix=prefix)
        for i in range(len(models_in_experiment) - 1):
            mod_1 = models_in_experiment[i]

            # the loaded embeddings and retrieval results of mod_1 (i) are reused for every mod_2 (j)
            mod_1.load_embs(dataset)
            r1, top_idx_1 = mod_1.run_retrieval(dataset)

            for j in range(i + 1, len(models_in_experiment)):
                mod_2 = models_in_experiment[j]

                print(f"\n{mod_1.id}-{mod_2.id}-{dataset.id}")
                mod_2.load_embs(dataset)
                r2, top_idx_2 = mod_2.run_retrieval(dataset)

                baselines = run_baselines(dataset, mod_1, mod_2,
                                          r1, r2, top_idx_1, top_idx_2)
                rows += baselines
                mod_1_weight = 0.5
                if args.tune:
                    for res_dict in baselines:
                        if "sim_score_softmax" in res_dict["run_id"]:
                            assert f"{mod_1.id}-{mod_2.id}" in res_dict["run_id"]
                            mod_1_weight = res_dict["weight"]

                exp_results = []
                input_params = [(lr, k, n, t, mixture_alpha, loss_func, optimizer, optimization_func)
                                for lr, k, n, t, mixture_alpha, loss_func, optimizer, optimization_func,
                                in product(lrs, ks, n_steps, Ts, mixture, loss_funcs, optimizers, optimization_funcs)]

                if args.tune and len(input_params) > 1:
                    mod_1_best_params = tune_hyper(dataset, mod_1, mod_2, device, input_params,
                                                   weight_for_feedback_model=1-mod_1_weight)
                    mod_2_best_params = tune_hyper(dataset, mod_2, mod_1, device, input_params,
                                                   weight_for_feedback_model=mod_1_weight)
                    input_params = [(dataset, mod_1, mod_2, device, *mod_1_best_params),
                                    (dataset, mod_2, mod_1, device, *mod_2_best_params)]
                else:
                    input_params = [(dataset, mod_1, mod_2, device, *params) for params in input_params] + \
                                    [(dataset, mod_2, mod_1, device, *params) for params in input_params]

                description = f"Running all query optimizations for pair {mod_1.id},{mod_2.id} (parallelization={args.use_parallelization})"
                if args.use_parallelization:
                    all_results = []
                    pbar = tqdm.tqdm(total=len(input_params), desc=description)
                    pool = ThreadPool(4) if on_ccc() else Pool(cpu_count())
                    for run_args in input_params:
                        all_results.append(
                            pool.apply_async(run_query_optimizations, run_args, callback=lambda _: pbar.update(1)))
                    pool.close()
                    pool.join()
                    for process_result in all_results:
                        exp_results.append(process_result.get())
                    pbar.close()
                else:
                    for run_args in tqdm.tqdm(input_params, desc=description):
                        exp_results.append(run_query_optimizations(*run_args))

                for result_dict, (_, _, _, _, lr, k, n, t, mixture_alpha, loss_func, optimizer, optimization_func) in zip(
                        exp_results, input_params):
                    if len(result_dict) == 0:
                        continue
                    rows.append({
                        **result_dict,
                        "weight": 0,
                        "lr": lr,
                        "k": k,
                        "n_steps": n,
                        "temp": t,
                        "mixture_alpha": mixture_alpha,
                        "loss_func": loss_func.__name__,
                        "optimization_func": optimization_func,
                        "optimizer": optimizer.__name__,
                        "dataset": dataset.id,
                    })

                if (j == (len(models_in_experiment) - 1)
                        and i < (len(models_in_experiment) - 2)
                        and (models_in_experiment[i + 1].id == mod_2.id or models_in_experiment[i + 2].id == mod_2.id)):
                    pass  # mod_2 embeddings can be reused
                else:
                    mod_2.clean_embs(dataset)

                gc.collect()
                torch.cuda.empty_cache()

            mod_1.clean_embs(dataset)

        # after every dataset, save all results collected so far
        df = pd.DataFrame(rows)
        metrics = sorted({k for m in df["metrics"] for k in m.keys()})
        for metric in metrics:
            tmp = df.copy()
            if metric == "raw_results":
                tmp["raw_results"] = tmp["metrics"].apply(lambda m: m.get(metric, np.nan))
                raw_results = defaultdict(defaultdict)
                for _, row in tmp.iterrows():
                    raw_results[row["run_id"]][row["dataset"]] = row.to_dict()
                with open(os.path.join(out_dir, "raw_results.json"), "w") as f:
                    json.dump(raw_results, f)
                continue

            def merge_values(x):
                non_nan = x.dropna().unique()
                if len(non_nan) > 1:
                    return non_nan.tolist()
                else:
                    return non_nan[0]

            id_cols = [col for col in df.columns if col not in {"dataset", "metrics"}]
            tmp["metric_value"] = tmp["metrics"].apply(lambda m: m.get(metric, np.nan))
            wide = tmp.pivot_table(index=id_cols, columns="dataset", values="metric_value", aggfunc="first").reset_index()
            if args.tune:
                os.makedirs(os.path.join(out_dir, "tune"), exist_ok=True)
                wide.to_csv(os.path.join(out_dir, "tune", f"{metric}.csv"), index=False)
                wide = wide.groupby("run_id", as_index=False).agg(merge_values)
            dataset_cols = [col for col in wide.columns if col not in id_cols]
            wide["average"] = wide[dataset_cols].mean(axis=1, numeric_only=True)
            wide.to_csv(os.path.join(out_dir, f"{metric}.csv"), index=False)
    return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', nargs='+', required=True)
    parser.add_argument('-p', '--use_parallelization', type=ast.literal_eval, default=False)
    parser.add_argument('-t', '--tune', type=ast.literal_eval, default=False)
    parser.add_argument('-o', '--out_dir_suffix', default='')

    args = parser.parse_args()
    main(args)
