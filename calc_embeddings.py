import argparse
import gc
import io

import torch
from natsort import natsorted, ns

from dataset_configs import RagDataset
from embedding_configs import Embedders, Modality, EncoderConfig


def load_documents(path):
    docs = []
    for child in natsorted(path.iterdir(), alg=ns.IGNORECASE):
        if not child.is_file():
            continue
        file_bytes = child.read_bytes()
        stream = io.BytesIO(file_bytes)
        docs.append(stream)
    return docs


def main(args):
    configs: list[EncoderConfig] = [getattr(Embedders, model) for model in args.models]

    with torch.no_grad():
        for config in configs:
            embedder = config.embedder()
            for dataset_name in args.datasets:
                dataset = RagDataset(dataset_name, prefix=args.datasets_path_prefix)
                print(f"Running for {config.model_id}")
                if config.modality == Modality.VISION:
                    folder = dataset.path / "images"
                    img_docs = natsorted(folder.iterdir(), alg=ns.IGNORECASE)
                    docs = [f for f in img_docs if f.is_file() and not f.name == ".DS_Store"]
                else:
                    folder = dataset.path / "texts"
                    text_docs = load_documents(folder)
                    docs = [c.read().decode() for c in text_docs]
                benchmark = dataset.get_benchmark_obj()
                queries = [q for q in benchmark['question']]
                embedder.calc_and_save_embeddings(dataset.path, docs, queries, config.model_id,
                                                  custom_out_path=args.out_path)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    from dataset_configs import REAL_MM_RAG_DATASETS, VIDORE1_DATASETS, VIDORE2_DATASETS
    parser = argparse.ArgumentParser()
    all_models = [x for x in Embedders.__dict__.keys() if not x.startswith("_")]
    parser.add_argument('--datasets', nargs='+', default=VIDORE2_DATASETS)
    parser.add_argument('--models', nargs='+', required=True, choices=all_models)
    parser.add_argument('--datasets_path_prefix', default="/proj/omri/")
    parser.add_argument('--out_path')

    main_args = parser.parse_args()
    main(main_args)
