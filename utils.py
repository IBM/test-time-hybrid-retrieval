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
    t = t.coalesce()

    original_rows = t.indices()[0]
    original_cols = t.indices()[1]
    original_vals = t.values()

    # Efficiently find which non-zero elements to keep and where they should go
    # in the new tensor. `matches` will be a boolean matrix of shape
    # (num_non_zero_elements, num_slice_indices).
    matches = original_rows.unsqueeze(1) == slice_indices.unsqueeze(0)

    # Get the coordinates of the matches. This gives us two tensors:
    #    - nnz_indices: The index of the non-zero element in the original tensor.
    #    - new_row_indices: The new row index for that element in the output tensor.
    nnz_indices, new_row_indices = matches.nonzero(as_tuple=True)

    # Select the corresponding column indices and values for our new tensor
    new_col_indices = original_cols[nnz_indices]
    new_values = original_vals[nnz_indices]
    new_indices = torch.stack([new_row_indices, new_col_indices])

    new_shape = (slice_indices.shape[0], t.shape[1])
    sliced_tensor = torch.sparse_coo_tensor(new_indices,
                                            new_values,
                                            size=new_shape).coalesce()
    return sliced_tensor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
