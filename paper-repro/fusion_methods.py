from collections import defaultdict

import numpy as np
import torch


def reciprocal_rank_fusion(r1, r2, alpha=0.5, k=60):
    r = {}
    for q in r1.keys():
        curr_r1 = r1[q]
        curr_r2 = r2[q]
        rrf_scores = defaultdict(int)
        for i, ((doc1, _), (doc2, _)) in enumerate(zip(curr_r1, curr_r2)):
            rank = i + 1
            rrf_scores[doc1] += 2*alpha / (k + rank)
            rrf_scores[doc2] += 2*(1-alpha) / (k + rank)
        r[q] = [(doc, score)
                for doc, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)][:len(curr_r1)]
    return r


def average_ranking_fusion(r1, r2, alpha=0.5):

    def get_rank(doc, curr_r):
        for i, (d, _) in enumerate(curr_r):
            if d == doc:
                return i + 1
        return len(curr_r) + 1

    r = {}
    for q in r1.keys():
        curr_r1 = r1[q]
        curr_r2 = r2[q]
        union = set([doc for doc, _ in curr_r1]).union(set([doc for doc, _ in curr_r2]))
        raw_rank_averages = {}
        for doc in union:
            avg_rank = alpha*get_rank(doc, curr_r1) + (1-alpha)*get_rank(doc, curr_r2)
            raw_rank_averages[doc] = avg_rank
        r[q] = [(doc, 1/(score+1e-6))
                for doc, score in sorted(raw_rank_averages.items(), key=lambda x: x[1])][:len(curr_r1)]
    return r


def normalize_softmax(rankings):
    scores = torch.from_numpy(np.array([score for _, score in rankings]))
    scores = torch.softmax(scores, dim=0).tolist()
    new_rankings = []
    for i, ranking in enumerate(rankings):
        new_rankings.append((ranking[0], scores[i]))
    return new_rankings


def normalize_min_max(rankings):
    s = torch.tensor([b for _, b in rankings], dtype=torch.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return [(a, v.item()) for (a, _), v in zip(rankings, s)]


def sim_score_fusion(r1, r2, normal_func=normalize_min_max, alpha=0.5):
    def get_score(doc, curr_r):
        for d, s in curr_r:
            if d == doc:
                return s
        return 0

    r = {}
    for q in r1.keys():
        curr_r1 = r1[q]
        curr_r2 = r2[q]
        union = set([doc for doc, _ in curr_r1]).union(set([doc for doc, _ in curr_r2]))
        curr_r1, curr_r2 = normal_func(curr_r1), normal_func(curr_r2)
        raw_score_averages = {}
        for doc in union:
            agg_score = alpha*get_score(doc, curr_r1) + (1-alpha)*get_score(doc, curr_r2)
            raw_score_averages[doc] = agg_score
        r[q] = [(doc, score)
                for doc, score in sorted(raw_score_averages.items(), key=lambda x: x[1], reverse=True)][:len(curr_r1)]
    return r
