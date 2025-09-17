import importlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from fastembed import SparseTextEmbedding
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel
from transformers.utils.import_utils import is_flash_attn_2_available

from utils import get_device, batch


class Embedder:
    max_length: int = None
    batch_size: int = None
    dtype: np.floating = None
    is_multi_vector: bool = False
    is_sparse: bool = False
    file_suffix = "_embeddings"

    def calc_and_save_embeddings(self, dataset_path: Path, docs: list[str], queries: list[str], model_id: str,
                                 custom_out_path=None):
        model_name = model_id.split("/")[1]
        if custom_out_path:
            out_path = Path(custom_out_path) / dataset_path.as_posix().split("/")[-1]
        else:
            out_path = dataset_path
        out_path = out_path / (model_name + self.file_suffix)
        out_path.mkdir(parents=True, exist_ok=True)

        docs_file = out_path / "docs.npz"
        queries_file = out_path / "queries.npz"

        # TODO check if the embeddings are already there
        print(f"Encoding documents - {model_id}")
        docs_embeddings, query_embeddings = self.calc_embeddings(model_id, docs, queries)
        print(f"Writing {len(docs_embeddings)} encoded documents and {len(query_embeddings)} encoded queries to disk")
        np.savez_compressed(queries_file, query_embeddings)
        np.savez_compressed(docs_file, docs_embeddings)

    def calc_embeddings(self, model_id: str, docs: list[str], queries: list[str]):
        raise NotImplementedError()


class JinaEmbedder(Embedder):
    dtype = np.float32

    def calc_embeddings(self, model_id, docs, queries):
        m = importlib.import_module("transformers.modeling_flash_attention_utils")

        try:
            from flash_attn.layers.rotary import apply_rotary_emb as _apply_rotary_emb
        except Exception:
            _apply_rotary_emb = None
        try:
            from flash_attn import flash_attn_varlen_func as _favf
        except Exception:
            _favf = None

        setattr(m, "apply_rotary_emb", getattr(m, "apply_rotary_emb", _apply_rotary_emb))
        setattr(m, "flash_attn_varlen_func", getattr(m, "flash_attn_varlen_func", _favf))

        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.to(get_device())

        with torch.no_grad():
            query_embeddings = model.encode_text(
                texts=queries,
                task="retrieval",
                prompt_name="query",
                return_multivector=self.is_multi_vector,
            )
            document_embeddings = self._encode_documents(model, docs)
        query_embeddings = self._convert_to_output_format(query_embeddings)
        document_embeddings = self._convert_to_output_format(document_embeddings)

        return document_embeddings, query_embeddings

    def _convert_to_output_format(self, embeddings):
        if self.is_multi_vector:
            if len(set(e.shape for e in embeddings)) > 1:  # embedding matrices have different shapes
                return np.array([e.cpu().numpy().astype(self.dtype) for e in embeddings], dtype=object)
            else:  # all embedding matrices have the same shape
                return np.array([e.cpu().numpy().astype(self.dtype) for e in embeddings])
        else:
            return [e.cpu() for e in embeddings]

    def _encode_documents(self, model, docs):
        raise NotImplementedError()


class JinaImageEmbedder(JinaEmbedder):
    def _encode_documents(self, model, docs):
        images = [Image.open(path) for path in docs]
        return model.encode_image(
            images=images,
            task="retrieval",
            return_multivector=self.is_multi_vector,
        )


class JinaTextEmbedder(JinaEmbedder):
    file_suffix = "_text_embeddings"

    def _encode_documents(self, model, docs):
        return model.encode_text(
            texts=docs,
            task="retrieval",
            prompt_name="passage",
            return_multivector=self.is_multi_vector,
        )


class JinaImageMultiEmbedder(JinaImageEmbedder):
    is_multi_vector = True
    file_suffix = "_multi_embeddings"


class JinaTextMultiEmbedder(JinaTextEmbedder):
    is_multi_vector = True
    file_suffix = "_multi_txt_embeddings"


class SentenceTransformersEmbedder(Embedder):
    query_extra_kwargs: dict
    doc_extra_kwargs: dict

    def calc_embeddings(self, model_id, docs, queries):
        device = get_device()
        model = SentenceTransformer(model_id)
        model.max_seq_length = self.max_length
        model.eval().half().to(device)

        query_embeddings = self._encode(queries, model, device, extra_kwargs=self.query_extra_kwargs)
        document_embeddings = self._encode(docs, model, device, extra_kwargs=self.doc_extra_kwargs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return document_embeddings, query_embeddings

    def _encode(self, texts: list[str], model: SentenceTransformer, device: str, extra_kwargs: dict = None):
        extra_kwargs = extra_kwargs if extra_kwargs else {}
        out = []
        for batch_texts in tqdm(batch(texts, self.batch_size),
                                desc=f"Calculating embeddings (Batch size={self.batch_size})",
                                total=math.ceil(len(texts) / self.batch_size)):
            with torch.inference_mode():
                embs = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=device,
                    **extra_kwargs,
                ).astype(self.dtype, copy=False)
            out.append(embs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)


class QwenEmbedder(SentenceTransformersEmbedder):
    max_length = 512
    batch_size = 8
    dtype = np.float32

    query_extra_kwargs = {"prompt_name": "query"}
    doc_extra_kwargs = {"prompt_name": None}


class LinqMistralEmbedder(SentenceTransformersEmbedder):
    max_length = 512
    batch_size = 8
    dtype = np.float32

    task = "Given a question, retrieve Wikipedia passages that answer the question"
    prompt = f"Instruct: {task}\nQuery: "

    query_extra_kwargs = {"prompt": prompt}
    doc_extra_kwargs = {"prompt": None}


class BatchedMultiEmbedder(Embedder):
    image_batch_size: int
    text_batch_size: int
    is_multi_vector = True
    file_suffix = "_multi_embeddings"

    @staticmethod
    def to_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    @staticmethod
    def pad_for_concat(arrs, pad_axis=1):
        if not arrs:
            return arrs
        target = max(a.shape[pad_axis] for a in arrs)
        out = []
        for a in arrs:
            if a.shape[pad_axis] == target:
                out.append(a)
            else:
                pad = [(0, 0)] * a.ndim
                pad[pad_axis] = (0, target - a.shape[pad_axis])
                out.append(np.pad(a, pad, mode="constant"))
        return out


class NvidiaEmbedder(BatchedMultiEmbedder):
    image_batch_size = 64
    text_batch_size = 256

    def calc_embeddings(self, model_id, docs, queries):
        model = AutoModel.from_pretrained(
            model_id,
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c'  # TODO what is this?
        ).eval()

        img_embs = []
        with torch.no_grad():
            for batch_paths in batch(docs, self.image_batch_size):
                imgs = [Image.open(p) for p in batch_paths]
                emb = model.forward_passages(imgs, batch_size=len(imgs))
                img_embs.extend([t.detach().to(torch.float32).cpu().numpy() for t in self.to_list(emb)])
                for im in imgs:
                    im.close()

        q_embs = []
        with torch.no_grad():
            for batch_q in batch(queries, self.text_batch_size):
                emb = model.forward_queries(batch_q, batch_size=len(batch_q))
                q_embs.extend([t.detach().to(torch.float32).cpu().numpy() for t in self.to_list(emb)])

        print("concatenate imgs")
        document_embeddings = np.concatenate(self.pad_for_concat(img_embs, pad_axis=1), axis=0)
        print("concatenate queries")
        query_embeddings = np.concatenate(self.pad_for_concat(q_embs, pad_axis=1), axis=0)

        return document_embeddings, query_embeddings


class NomicEmbedder(BatchedMultiEmbedder):
    image_batch_size = 32
    text_batch_size = 256

    def calc_embeddings(self, model_id, docs, queries):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        model = ColQwen2_5.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_id)

        img_embs = []
        with torch.no_grad():
            for batch_paths in batch(docs, self.image_batch_size):
                imgs = [Image.open(p) for p in batch_paths]
                processed_images = processor.process_images(imgs).to(model.device)
                emb = model(**processed_images)
                img_embs.extend([t.detach().to(torch.float32).cpu().numpy() for t in self.to_list(emb)])
                for im in imgs:
                    im.close()

        q_embs = []
        with torch.no_grad():
            for batch_q in batch(queries, self.text_batch_size):
                processed_queries = processor.process_queries(batch_q).to(model.device)
                emb = model(**processed_queries)
                q_embs.extend([t.detach().to(torch.float32).cpu().numpy() for t in self.to_list(emb)])

        document_embeddings = np.concatenate(self.pad_for_concat(img_embs, pad_axis=1), axis=0)
        query_embeddings = np.concatenate(self.pad_for_concat(q_embs, pad_axis=1), axis=0)

        return document_embeddings, query_embeddings


class SparseEmbedder(Embedder):
    is_sparse = True

    def calc_embeddings(self, model_id: str, docs: list[str], queries: list[str]):
        model = SparseTextEmbedding(model_name=model_id)
        document_embeddings = list(model.embed(docs))
        query_embeddings = list(model.embed(queries))
        return document_embeddings, query_embeddings


class Modality(Enum):
    TEXT = "text"
    VISION = "vision"


@dataclass
class EncoderConfig:
    model_id: str
    modality: Modality
    embedder: type[Embedder]


class Embedders:
    qwen_text = EncoderConfig(model_id="Qwen/Qwen3-Embedding-4B", modality=Modality.TEXT,
                              embedder=QwenEmbedder)
    linq = EncoderConfig(model_id="Linq-AI-Research/Linq-Embed-Mistral", modality=Modality.TEXT,
                         embedder=LinqMistralEmbedder)
    nvidia = EncoderConfig(model_id="nvidia/llama-nemoretriever-colembed-3b-v1", modality=Modality.VISION,
                           embedder=NvidiaEmbedder)
    jina_multi = EncoderConfig(model_id="jinaai/jina-embeddings-v4", modality=Modality.VISION,
                               embedder=JinaImageMultiEmbedder)
    jina_single = EncoderConfig(model_id="jinaai/jina-embeddings-v4", modality=Modality.VISION,
                                embedder=JinaImageEmbedder)
    jina_text_multi = EncoderConfig(model_id="jinaai/jina-embeddings-v4", modality=Modality.TEXT,
                                    embedder=JinaTextMultiEmbedder)
    jina_text_single = EncoderConfig(model_id="jinaai/jina-embeddings-v4", modality=Modality.TEXT,
                                     embedder=JinaTextEmbedder)
    colnomic = EncoderConfig(model_id="nomic-ai/colnomic-embed-multimodal-7b", modality=Modality.VISION,
                             embedder=NomicEmbedder)
    bm25 = EncoderConfig(model_id="Qdrant/bm25", modality=Modality.TEXT,
                         embedder=SparseEmbedder)


all_embedders = [x for x in Embedders.__dict__.keys() if not x.startswith("_")]
