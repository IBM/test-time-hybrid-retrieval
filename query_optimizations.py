from enum import Enum
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

from dataset_configs import DataSplit
from retriever import get_topk, Retriever
from utils import get_device, slice_sparse_coo_tensor


def js_divergence(p, q, eps=1e-8):
    m = .5 * (p + q)
    return .5 * (torch.sum(p * torch.log((p + eps) / (m + eps))) + torch.sum(q * torch.log((q + eps) / (m + eps))))


def kl_divergence(p, q, eps=1e-8):
    return torch.sum(p * torch.log((p + eps) / (q + eps)))


def optimize_queries_with_search(main_model: Retriever, feedback_model: Retriever, dataset, device, split: DataSplit,
                                 optimize_queries_func: Callable, **kwargs):
    optimized_queries, _, _ = optimize_queries_func(
        main_model=main_model, feedback_model=feedback_model, dataset=dataset, device=device, split=split, **kwargs)
    r, _ = main_model.run_retrieval(dataset, q=optimized_queries.to(device), split=split)
    return r


def optimize_queries_no_search(main_model: Retriever, feedback_model: Retriever, dataset, device, split: DataSplit,
                               optimize_queries_func: Callable, **kwargs):
    optimized_queries, doc_indices_per_query, docs_per_query = optimize_queries_func(
        main_model=main_model, feedback_model=feedback_model, dataset=dataset, device=device, split=split, **kwargs)

    questions = dataset.benchmark['question']
    q_indices = dataset.get_queries_indices(split)
    questions = questions[q_indices.numpy()]
    r = {}
    for question, optimized_query, doc_indices, docs in zip(
            questions, optimized_queries, doc_indices_per_query, docs_per_query):
        scores = main_model.compute_scores(optimized_query.unsqueeze(0), docs)
        r[question] = [(idx, val) for idx, val in zip(doc_indices.tolist(), scores.tolist())]
    return r


def optimize_queries_main(main_model, feedback_model, dataset, k, lr, n_steps, T, mixture_alpha, loss_func, device,
                          optimizer, split: DataSplit):
    f_d, f_q_dict = feedback_model.load_embs(dataset)
    f_q = f_q_dict[split]
    d, q = main_model.d.to(device), main_model.q_dict[split].to(device)
    Q = q.size(0)
    out_q = []
    doc_indices_per_query = []
    docs_per_query = []

    qs = [q[i].unsqueeze(0).clone().detach().requires_grad_(True) for i in range(Q)]
    opt = optimizer(qs, lr=lr)  # one optimizer for all queries
    top_k = get_topk(d, q, k, is_multi=main_model.is_multi)[1].squeeze()
    for q_id in range(Q):
        q1 = qs[q_id]
        q2 = f_q[q_id].unsqueeze(0).to(device)
        set1 = d[top_k[q_id]]
        doc_indices_per_query.append(top_k[q_id])
        docs_per_query.append(set1)

        if f_d.is_sparse:
            set2 = slice_sparse_coo_tensor(f_d, slice_indices=top_k[q_id]).to(device)
        else:
            set2 = f_d[(top_k[q_id]).to(f_d.device)].to(device)
        d2 = F.softmax(feedback_model.compute_scores(q2, set2) / T, -1).to(device)
        d1 = F.softmax(main_model.compute_scores(q1, set1) / T, -1).to(device)
        mixture_d = (1-mixture_alpha)*d1 + mixture_alpha*d2
        for step in range(n_steps):
            d1 = F.softmax(main_model.compute_scores(q1, set1) / T, -1).to(device)
            loss = loss_func(mixture_d, d1)
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            opt.step()
        out_q.append(q1.detach().squeeze())

    return torch.stack(out_q), doc_indices_per_query, docs_per_query


def optimize_queries_union(main_model, feedback_model, dataset, k, lr, n_steps, T, mixture_alpha, loss_func, device,
                           optimizer, split: DataSplit):
    f_d, f_q_dict = feedback_model.load_embs(dataset)
    d, q, f_d, f_q = main_model.d.to(device), main_model.q_dict[split].to(device), f_d.to(device), f_q_dict[split].to(device)
    Q = q.size(0)
    out_q = []
    doc_indices_per_query = []
    docs_per_query = []

    qs = [q[i].unsqueeze(0).clone().detach().requires_grad_(True) for i in range(Q)]
    opt = optimizer(qs, lr=lr)  # one optimizer for all queries

    sim1 = main_model.compute_scores(q, d)
    sim2 = feedback_model.compute_scores(f_q, f_d)

    _, top_idx1 = torch.topk(sim1, k=k, dim=-1)
    _, top_idx2 = torch.topk(sim2, k=k, dim=-1)

    for i in range(Q):
        u = torch.unique(torch.cat([top_idx1[i], top_idx2[i]]))
        docs_u = d.index_select(0, u.to(d.device))
        doc_indices_per_query.append(u)
        docs_per_query.append(docs_u)

        d1 = torch.softmax(sim1[i, u] / T, dim=-1).to(device)
        d2 = torch.softmax(sim2[i, u] / T, dim=-1).to(device)
        mixture_d = (1-mixture_alpha)*d1 + mixture_alpha*d2
        for step in range(n_steps):
            d1 = F.softmax(main_model.compute_scores(qs[i], docs_u) / T, -1).to(device)
            loss = loss_func(mixture_d, d1)
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            opt.step()
        out_q.append(qs[i].detach().squeeze())

    return torch.stack(out_q), doc_indices_per_query, docs_per_query


def optimize_queries_union_sample(main_model, feedback_model, dataset, k, lr, n_steps, T, mixture_alpha, loss_func, device,
                                  optimizer, split):
    f_d, f_q_dict = feedback_model.load_embs(dataset)
    d, q, f_d, f_q = main_model.d.to(device), main_model.q_dict[split].to(device), f_d.to(device), f_q_dict[split].to(device)
    Q = q.size(0)
    out_q = []
    doc_indices_per_query = []
    docs_per_query = []

    qs = [q[i].unsqueeze(0).clone().detach().requires_grad_(True) for i in range(Q)]
    opt = optimizer(qs, lr=lr)  # one optimizer for all queries

    sim1 = main_model.compute_scores(q, d)
    sim2 = feedback_model.compute_scores(f_q, f_d)

    _, top_idx1 = torch.topk(sim1, k=k, dim=-1)
    _, top_idx2 = torch.topk(sim2, k=k, dim=-1)

    for i in range(Q):
        u = torch.unique(torch.cat([top_idx1[i], top_idx2[i]]))
        docs_u = d.index_select(0, u.to(d.device))
        doc_indices_per_query.append(u)
        docs_per_query.append(docs_u)

        d1 = sim1[i, u].to(device)
        d2 = sim2[i, u].to(device)
        mixture_d = (1-mixture_alpha)*d1 + mixture_alpha*d2
        for step in range(n_steps):
            indices_to_sample = torch.randperm(u.size()[0])[:10]
            sample_docs_u = d.index_select(0, u[indices_to_sample].to(d.device))
            d1 = F.softmax(main_model.compute_scores(qs[i], sample_docs_u) / T, -1).to(device)
            other_d = (mixture_d[indices_to_sample] / T).softmax(-1)
            loss = loss_func(other_d, d1)
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            opt.step()
        out_q.append(qs[i].detach().squeeze())
    return torch.stack(out_q), doc_indices_per_query, docs_per_query


def optimize_queries_dynamic(mod_1, mod_2, dataset, k, lr, n_steps, T, mixture_alpha, loss_func):
    # TODO not functional currently
    device = get_device()
    for _ in range(5):
        optimized_queries_1 = mod_1.optimize_queries(mod_2, dataset, k=k, lr=lr, n_steps=n_steps, T=T,
                                                     mixture_alpha=mixture_alpha, loss_func=loss_func)
        r3, _ = mod_1.run_retrieval(dataset, q=optimized_queries_1.to(device))
        print(dataset.evaluate(r3))
        mod_1.set_queries(optimized_queries_1)

        optimized_queries_2 = mod_2.optimize_queries(mod_1, dataset, k=k, lr=lr, n_steps=n_steps, T=T,
                                                     mixture_alpha=mixture_alpha, loss_func=loss_func)
        r4, _ = mod_2.run_retrieval(dataset, q=optimized_queries_2.to(device))
        print(dataset.evaluate(r4))
        mod_2.set_queries(optimized_queries_2)


class OptimizationFunctions(Enum):
    main_with_search = partial(optimize_queries_with_search,
                               optimize_queries_func=optimize_queries_main)
    union_with_search = partial(optimize_queries_with_search,
                                optimize_queries_func=optimize_queries_union)
    union_sample_with_search = partial(optimize_queries_with_search,
                                       optimize_queries_func=optimize_queries_union_sample)

    main_no_search = partial(optimize_queries_no_search,
                             optimize_queries_func=optimize_queries_main)
    union_no_search = partial(optimize_queries_no_search,
                              optimize_queries_func=optimize_queries_union)
    union_sample_no_search = partial(optimize_queries_no_search,
                                     optimize_queries_func=optimize_queries_union_sample)


def scores_feedback(main_model, feedback_model, dataset, top_k_idxs, alpha=0.5, split=DataSplit.TEST):
    device = get_device()
    f_d, f_q_dict = feedback_model.load_embs(dataset)
    d, q, f_d, f_q = main_model.d.to(device), main_model.q_dict[split].to(device), f_d.to(device), f_q_dict[split].to(device)
    top_k = top_k_idxs.squeeze()
    results = {}
    questions = dataset.benchmark['question']
    q_indices = dataset.get_queries_indices(split)
    questions = questions[q_indices.numpy()]
    for q_id, question in enumerate(questions):
        q1 = q[q_id].unsqueeze(0)
        q2 = f_q[q_id].unsqueeze(0)
        set1 = d[top_k[q_id]]
        if f_d.is_sparse:
            set2 = slice_sparse_coo_tensor(f_d, slice_indices=top_k[q_id])
        else:
            set2 = f_d[top_k[q_id]]
        d2 = F.softmax(feedback_model.compute_scores(q2, set2), -1).to(device)
        d1 = F.softmax(main_model.compute_scores(q1, set1), -1).to(device)
        mixture_d = alpha * d1 + (1 - alpha) * d2
        results[question] = sorted([(idx.item(), val.item()) for (idx, val) in zip(top_k[q_id], mixture_d)],
                                   key=lambda x: x[1], reverse=True)

    return results
