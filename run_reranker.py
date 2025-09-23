import argparse
from collections import defaultdict
from PIL import Image
from transformers import AutoModel
from multiprocessing import Pool, cpu_count
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import gc
from dataset_configs import Datasets, RagDataset
from embedding_configs import Embedders
from retriever import Retriever
from utils import get_device, set_seed, on_ccc
from dataset_configs import DataSplit
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import time
from bench_latency_vidore2 import sync_accel
from pathlib import Path
def main(args):
    device = get_device()
    prefix = "/proj/omri/" if on_ccc() else ""

    vidore2 = [
        # Datasets.esg_reports_v2,
        # Datasets.biomedical_lectures_v2,
        # Datasets.economics_reports_v2,
        Datasets.esg_reports_human_labeled_v2
    ]

    models_in_experiment = [
        Embedders.colnomic,
    ]

    datasets_in_experiment = [*vidore2]

    ks = [
        5,
        10,
        20
    ]

    max_pixels= 2048*28*28

    # Load processor and model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                        max_pixels=max_pixels)
    reranker_qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        "lightonai/MonoQwen2-VL-v0.1",
        # device_map="auto",
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device)


    models_in_experiment = [Retriever(m) for m in models_in_experiment]
    out_dir = f"output/results-{args.out_dir_suffix}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results in {out_dir}")


    latencies = defaultdict(list)
    rows = []
    for k in ks:
        for dataset_name in datasets_in_experiment:
            set_seed()
            dataset = RagDataset(dataset_name, prefix=prefix)
            q_idx = dataset.get_queries_indices(split=DataSplit.DEV)
            questions_all = dataset.get_benchmark_obj()['question']
            queries = questions_all.iloc[q_idx].reset_index(drop=True)
            for i in range(len(models_in_experiment)):
                mod_1 = models_in_experiment[i]

                mod_1.load_embs(dataset)
                r1, top_idx_1 = mod_1.run_retrieval(dataset,split=DataSplit.DEV,k=k)
                info_dict = {
                    "lr": 0.0, "k": k, "n_steps": 0, "temp": 0.0,
                    "mixture_alpha": 0, "loss_func": "baseline",
                    "weight": 0,
                    "optimization_func": "", "optimizer": "",
                    "dataset": dataset.id,
                }
                rows.append({
                **info_dict,
                "run_id": mod_1.id,
                "main_model": mod_1.id,
                "feedback_model": "",
                "metrics": dataset.evaluate(r1)
            })
                mod_1.clean_embs(dataset)
                gc.collect()
                torch.cuda.empty_cache()
                r = {}
                for q_id in tqdm(range(top_idx_1.size(0))):
                    t0 = time.perf_counter()
                    
                    query = queries[q_id]
                    # Construct the prompt and prepare input
                    prompt = (
                        "Assert the relevance of the previous image document to the following query, "
                        "answer True or False. The query is: {query}"
                    ).format(query=query)    
                    docs_id = top_idx_1[q_id][:k]
                    docs = [f"{dataset_name}/images/{doc_id}.jpg" for doc_id in docs_id]
                    images = [Image.open(doc) for doc in docs]
                    scores = []
                    for image in images:
                        messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                        ]
                        # Apply chat template and tokenize
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=text, images=image, return_tensors="pt").to(device)

                        # Run inference to obtain logits
                        with torch.no_grad():
                            outputs = reranker_qwen(**inputs)
                            logits_for_last_token = outputs.logits[:, -1, :]

                        # Convert tokens and calculate relevance score
                        true_token_id = processor.tokenizer.convert_tokens_to_ids("True")
                        false_token_id = processor.tokenizer.convert_tokens_to_ids("False")
                        relevance_score = torch.softmax(logits_for_last_token[:, [true_token_id, false_token_id]], dim=-1)

                        # Extract and display probabilities
                        true_prob = relevance_score[0, 0].item()
                        scores.append(true_prob)
                    
                    sync_accel()
                    dt = (time.perf_counter() - t0)*1000
                    latencies[f"{mod_1.id}-renaker-MonoQwen2-VL-v0.1-k-{k}"].append(dt)
                    q_text = str(queries.iloc[q_id] if hasattr(queries, "iloc") else queries[q_id])
                    pairs = list(zip(top_idx_1[q_id][:k].tolist(), scores))
                    pairs.sort(key=lambda t: t[1], reverse=True)
                    r[q_text] = [(int(doc_id), float(score)) for doc_id, score in pairs]

                rows.append({
                    **info_dict,
                    "run_id": f"{mod_1.id}-renaker-MonoQwen2-VL-v0.1-k-{k}",
                    "main_model": mod_1.id,
                    "feedback_model": "",
                    "metrics": dataset.evaluate(r)
                })


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

            id_cols = [col for col in df.columns if col not in {"dataset", "metrics"}]
            tmp["metric_value"] = tmp["metrics"].apply(lambda m: m.get(metric, np.nan))
            wide = tmp.pivot_table(index=id_cols, columns="dataset", values="metric_value", aggfunc="first").reset_index()
            dataset_cols = [col for col in wide.columns if col not in id_cols]
            wide["average"] = wide[dataset_cols].mean(axis=1, numeric_only=True)
            wide.to_csv(os.path.join(out_dir, f"{metric}.csv"), index=False)
    out_path = Path(os.path.join(out_dir, f"latency.csv"))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(latencies, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir_suffix', default='')

    args = parser.parse_args()
    main(args)
