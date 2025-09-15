import numpy as np
import torch


def reciprocal_rank_fusion(r1, r2, k=60):
    r = {}
    for q in r1.keys():
        curr_r1 = r1[q]
        curr_r2 = r2[q]
        rrf_scores = {}
        for i, ((doc1, _), (doc2, _)) in enumerate(zip(curr_r1, curr_r2)):
            rrf_scores[doc1] = rrf_scores.get(doc1, 0) + 1 / (k + i)
            rrf_scores[doc2] = rrf_scores.get(doc2, 0) + 1 / (k + i)
        r[q] = [(doc, score)
                for doc, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)][:len(curr_r1)]
    return r


def average_ranking_fusion(r1, r2):

    def get_rank(doc, curr_r):
        for i, (d, s) in enumerate(curr_r):
            if d == doc:
                return i
        return len(curr_r)

    r = {}
    for q in r1.keys():
        curr_r1 = r1[q]
        curr_r2 = r2[q]
        union = set([doc for doc, _ in curr_r1]).union(set([doc for doc, _ in curr_r2]))
        raw_rank_averages = {}
        for doc in union:
            avg_rank = (get_rank(doc, curr_r1) + get_rank(doc, curr_r2))/2
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


# def oracle_fusion(rankings_1, rankings_2, question, top_k=40):
#     gold_truth_ids = question["ground_truths_context_ids"]
#     ndcg1 = ndcg_at_k([vars(r) for r in rankings_1],gold_truth_ids)
#     ndcg2 = ndcg_at_k([vars(r) for r in rankings_2],gold_truth_ids)
#     if ndcg1 > ndcg2:
#         return [r.metadata["document_id"] for r in rankings_1]
#     return [r.metadata["document_id"] for r in rankings_2]

# def fuse_rankings(rag_results, model1_results, model2_results, fusion_func, fusion_factor=200, trimm_factor=200):
#     q_ids = set(rag_results.get_qids())
#     for q_id in q_ids:
#         context_model1 = model1_results[q_id]["context"][:fusion_factor]
#         context_model2 = model2_results[q_id]["context"][:fusion_factor]
#         if fusion_func == oracle_fusion:
#             docs_ids = oracle_fusion(context_model1, context_model2, rag_results[q_id])
#         else:
#             docs_ids = fusion_func(context_model1, context_model2, top_k=trimm_factor)
#         unique_docs = []
#         seen_ids = set()
#         for doc_id in docs_ids:
#             doc = get_doc(context_model1, context_model2, doc_id)
#             if doc_id not in seen_ids:
#                 seen_ids.add(doc_id)
#                 unique_docs.append(doc)
#
#         rag_results[q_id]["context"] = unique_docs
