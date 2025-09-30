import argparse
import gc
import json
import os
import time

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

from bench_latency_vidore2 import sync_accel
from dataset_configs import Datasets, RagDataset, DataSplit
from embedding_configs import Embedders
from retriever import Retriever
from utils import get_device, set_seed


class Reranker:
    model_id: str
    
    model = None
    processor = None


    def __init__(self):
        self.id = self.model_id.split("/")[-1]

    def load_model(self):
        raise NotImplementedError()

    def score_document_relevance(self, query: str, doc: str):
        raise NotImplementedError()


class MonoQwenReranker(Reranker):
    model_id = "lightonai/MonoQwen2-VL-v0.1"
    max_pixels = 2048 * 28 * 28

    prompt_template = (
        "Assert the relevance of the previous image document to the following query, "
        "answer True or False. The query is: {query}"
    )

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                       max_pixels=self.max_pixels)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=get_device(),
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            torch_dtype=torch.bfloat16,
        )

    def score_document_relevance(self, query: str, doc: str):
        device = get_device()

        image = Image.open(doc)
        prompt = self.prompt_template.format(query=query)
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
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(device)

        # Run inference to obtain logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_for_last_token = outputs.logits[:, -1, :]

        # Convert tokens and calculate relevance score
        true_token_id = self.processor.tokenizer.convert_tokens_to_ids("True")
        false_token_id = self.processor.tokenizer.convert_tokens_to_ids("False")
        relevance_score = torch.softmax(logits_for_last_token[:, [true_token_id, false_token_id]], dim=-1)

        # Extract and display probabilities
        true_prob = relevance_score[0, 0].item()
        return true_prob


def main(args):
    split = DataSplit.TEST

    vidore2 = [
        Datasets.esg_reports_v2,
        Datasets.biomedical_lectures_v2,
        Datasets.economics_reports_v2,
        Datasets.esg_reports_human_labeled_v2
    ]

    models_in_experiment = [
        Embedders.nvidia,
        Embedders.jina_multi,
        Embedders.colnomic,
    ]

    datasets_in_experiment = vidore2

    ks = [
        5,
        10,
        20
    ]

    reranker = MonoQwenReranker()
    reranker.load_model()

    models_in_experiment = [Retriever(m) for m in models_in_experiment]
    out_dir = f"output/results-{args.out_dir_suffix}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results in {out_dir}")

    latencies = defaultdict(list)
    rows = []
    for dataset_name in datasets_in_experiment:
        set_seed()
        dataset = RagDataset(dataset_name, prefix=args.datasets_path_prefix)
        q_idx = dataset.get_queries_indices(split=split)
        questions_all = dataset.get_benchmark_obj()['question']
        queries = questions_all.iloc[q_idx].reset_index(drop=True)
        for i in range(len(models_in_experiment)):
            mod_1 = models_in_experiment[i]

            mod_1.load_embs(dataset)
            r1, top_idx_1 = mod_1.run_retrieval(dataset, split=split, k=max(ks))
            info_dict = {
                "lr": 0.0, "k": 0, "n_steps": 0, "temp": 0.0,
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
            
            for k in ks:
                r = {}
                for q_id in tqdm(range(top_idx_1.size(0)), desc=f"reranking with {reranker.id} and k={k}"):
                    t0 = time.perf_counter()
                    
                    query = queries[q_id]
                    doc_ids_to_rerank = top_idx_1[q_id][:k]
                    scores = []
                    for doc_id in doc_ids_to_rerank:
                        doc = f"{args.datasets_path_prefix}{dataset_name}/images/{doc_id}.jpg"
                        score = reranker.score_document_relevance(query, doc)
                        scores.append(score)
                    
                    sync_accel()
                    dt = (time.perf_counter() - t0)*1000
                    latencies[f"{mod_1.id}-reranker-{reranker.id}-k-{k}"].append(dt)
                    doc_ids_and_scores = sorted(zip(doc_ids_to_rerank.tolist(), scores),
                                                key=lambda t: t[1], reverse=True)
                    r[query] = [(int(doc_id), float(score)) for doc_id, score in doc_ids_and_scores]

                rows.append({
                    **info_dict,
                    "run_id": f"{mod_1.id}-reranker-{reranker.id}-k-{k}",
                    "main_model": mod_1.id,
                    "feedback_model": "",
                    "k": k,
                    "metrics": dataset.evaluate(r)
                })

                # after every k, save all results collected so far
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

                latency_out_path = Path(os.path.join(out_dir, "latency.json"))
                with latency_out_path.open("w", encoding="utf-8") as f:
                    json.dump(latencies, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path_prefix', default='')
    parser.add_argument('-o', '--out_dir_suffix', default='reranking')

    args = parser.parse_args()
    main(args)