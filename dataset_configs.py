import ast
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval
import torch


class DataSplit(Enum):
    TEST = "test"
    DEV = "dev"


class Datasets:
    arxivqa = "Vidore1/arxivqa_test_subsampled_beir"
    docvqa = "Vidore1/docvqa_test_subsampled_beir"
    infovqa = "Vidore1/infovqa_test_subsampled_beir"
    tabfquad = "Vidore1/tabfquad_test_subsampled_beir"
    tatdqa = "Vidore1/tatdqa_test_beir"
    shiftproject = "Vidore1/shiftproject_test_beir"
    artificial_intelligence = "Vidore1/syntheticDocQA_artificial_intelligence_test_beir"
    energy_test = "Vidore1/syntheticDocQA_energy_test_beir"
    government_reports = "Vidore1/syntheticDocQA_government_reports_test_beir"
    healthcare = "Vidore1/syntheticDocQA_healthcare_industry_test_beir"

    esg_reports_v2 = 'Vidore2/esg_reports_v2'
    biomedical_lectures_v2 = 'Vidore2/biomedical_lectures_v2'
    economics_reports_v2 = 'Vidore2/economics_reports_v2'
    esg_reports_human_labeled_v2 = 'Vidore2/esg_reports_human_labeled_v2'

    FinReport = 'REAL-MM-RAG/FinReport'
    FinSlides = 'REAL-MM-RAG/FinSlides'
    TechReport = 'REAL-MM-RAG/TechReport'
    TechSlides = 'REAL-MM-RAG/TechSlides/'




VIDORE1_DATASETS = [
    Datasets.arxivqa,
    Datasets.docvqa,
    Datasets.infovqa,
    Datasets.tabfquad,
    Datasets.tatdqa,
    Datasets.shiftproject, 
    Datasets.artificial_intelligence,
    Datasets.energy_test,
    Datasets.government_reports,
    Datasets.healthcare
]


VIDORE2_DATASETS = [
    Datasets.esg_reports_v2,
    Datasets.biomedical_lectures_v2,
    Datasets.economics_reports_v2,
    Datasets.esg_reports_human_labeled_v2
]


REAL_MM_RAG_DATASETS = [
    Datasets.FinReport,
    Datasets.FinSlides,
    Datasets.TechReport,
    Datasets.TechSlides
]


class RagDataset:

    def __init__(self, path, prefix, dev_size=0.1):
        self.path = Path(prefix + path)
        self.id = self.path.stem
        self.dev_size = dev_size

        self.idx = None

        self.benchmark = self.get_benchmark_obj()
        self.num_queries = len(self.benchmark)

    def get_benchmark_obj(self) -> pd.DataFrame:
        csv_fp = self.path / "benchmark" / "benchmark.csv"
        df = pd.read_csv(
            csv_fp,
            converters={"correct_answer_document_ids": ast.literal_eval},
        )
        return df

    def get_queries_indices(self, split=DataSplit.TEST):
        if self.idx is None:
            self.idx = torch.randperm(self.num_queries)
        n = int(self.num_queries * self.dev_size)
        return self.idx[n:] if split == DataSplit.TEST else self.idx[:n]

    def evaluate(self, results, return_raw_results=True):
        b = self.benchmark[self.benchmark['question'].isin(list(results.keys()))]

        qs = list(b['question'])

        def _qrels_entry(rels):
            return {str(d): v for d, v in rels.items()}

        qrels = {str(q): _qrels_entry(rels) for q, rels in zip(b['question'], b['correct_answer_document_ids'])}
        run = {str(q): {str(d): float(s) for d, s in results[q]} for q in qs}
        ks = (1, 5, 10, 100)
        metrics = {f'ndcg_cut.{",".join(map(str, ks))}', f'recall.{",".join(map(str, ks))}'}
        scores = pytrec_eval.RelevanceEvaluator(qrels, metrics).evaluate(run)

        out = {}
        for k in ks:
            ndcg = [scores.get(str(q), {}).get(f'ndcg_cut_{k}', np.nan) for q in qs]
            rec = [scores.get(str(q), {}).get(f'recall_{k}', np.nan) for q in qs]
            out[f'ndcg@{k}'] = round(float(np.nanmean(ndcg)), 3)
            out[f'recall@{k}'] = round(float(np.nanmean(rec)), 3)
        if return_raw_results:
            out["raw_results"] = run
        return out
