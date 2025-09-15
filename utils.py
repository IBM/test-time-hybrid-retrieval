import os
import random

import numpy as np
import torch
import hashlib
import json


def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # mac GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def on_ccc():
    return os.path.isdir("/proj/")


def get_doc(context_1, context_2, doc_id):
    for doc in context_1:
        if doc.metadata["document_id"] == doc_id:
            return doc
    for doc in context_2:
        if doc.metadata["document_id"] == doc_id:
            return doc


def batch(x, n):
    for i in range(0, len(x), n):
        yield x[i:i + n]


def get_run_hash(*args):
    s = json.dumps(args, sort_keys=True, default=str)
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def slice_sparse_coo_tensor(t: torch.Tensor, slice_indices: torch.Tensor):
    def ainb(a, b):
        indices = torch.zeros_like(a, dtype=torch.uint8)
        for elem in b:
            indices = indices | (a == elem)

        return indices.type(torch.bool)

    new_shape = (slice_indices.shape[0], t.shape[1])
    t = t.coalesce()
    mask = ainb(a=t.indices()[0], b=slice_indices.to(t.device))
    sliced_tensor = torch.sparse_coo_tensor(t.indices()[:, mask],
                                            t.values()[mask],
                                            size=new_shape)
    return sliced_tensor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
