import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2  # only for cv2.error in the try/except
import datasets
import pandas as pd
from docling.document_converter import DocumentConverter
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm

SHRT_MAX = 32_767
SIDE_LIMIT = 32_000  # first‑stage clamp
DIAG_LIMIT = SHRT_MAX - 200  # ≈ 32500, with a small cushion
DOWNSAMPLE_FILTER = Image.LANCZOS
MAX_RETRIES = 3


def pil_image_to_temp_path(img: Image.Image, suffix=".jpg") -> Path:
    img = img.convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        img.save(tmp, format="JPEG", quality=92, optimize=True)
        name = tmp.name
    return Path(name)


def safe_resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > SIDE_LIMIT:
        scale = SIDE_LIMIT / float(max(w, h))
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), DOWNSAMPLE_FILTER)
    diag = math.hypot(w, h)
    if diag > DIAG_LIMIT:
        scale = DIAG_LIMIT / diag
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), DOWNSAMPLE_FILTER)
    return img.convert("RGB")


def run_docling_on_hf(dataset: str, name: str):
    converter = DocumentConverter()
    ds = datasets.load_dataset(dataset, name=name)["test"]
    images = {r["corpus_id"]: r["image"] for r in ds}

    out_dir = Path(dataset.split("/")[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid, img in tqdm(images.items()):
        md_path = out_dir / f"{cid}.md"
        if md_path.exists():
            continue  # already done

        img = safe_resize(img)

        # -------- robust conversion with up‑to‑3 automatic retries ------
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                tmp_path = pil_image_to_temp_path(img)
                result = converter.convert(tmp_path)
                break  # success
            except cv2.error as e:
                if "SHRT_MAX" not in str(e) or attempt == MAX_RETRIES:
                    raise  # different error or out of retries
                # shrink more and retry
                w, h = img.size
                img = img.resize((int(w * 0.8), int(h * 0.8)),
                                 DOWNSAMPLE_FILTER)

        with md_path.open("w") as f:
            f.writelines(result.document.export_to_markdown())


def download_hf_queries(dataset: str, out_dir: Path):
    for config_name in ["queries"]:
        try:
            ds = datasets.load_dataset(dataset, name=config_name,split="test")
        except:
            continue
    queries = {r["query_id"]: r["query"] for r in ds}

    qrels = datasets.load_dataset(dataset, name="qrels")["test"]
    query_to_gold = defaultdict(dict)
    for e in qrels:
        query = queries[e['query_id']]
        query_to_gold[query].update({e['corpus_id']: e['score']})

    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.DataFrame([{"question": q, "correct_answer_document_ids": qrel}
                                for q, qrel in query_to_gold.items()])
    benchmark_df.to_csv(out_dir / "benchmark.csv", index=False)


def download_hf_images(dataset: str, name: str, out_dir: Path):
    for config_name in ["corpus"]:
        try:
            ds = datasets.load_dataset(dataset, name=config_name,split="test")
        except:
            continue
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(list(out_dir.rglob("*.jpg"))) == len(ds):
        return

    for record in tqdm(ds, desc="saving images"):
        cid, img = record["corpus_id"], record["image"]
        img_path = out_dir / f"{cid}.jpg"
        if img_path.exists():
            continue
        img.save(img_path, format="JPEG")

def download_hf_markdown(dataset: str, name: str, out_dir: Path):
    # Load the HF split that contains the markdown
    for config_name in ["corpus"]:
        try:
            ds = datasets.load_dataset(dataset, name=config_name, split="test")
        except:
            continue

    out_dir.mkdir(parents=True, exist_ok=True)

    for record in tqdm(ds, desc="saving markdown"):
        cid = record["corpus_id"]
        md  = record.get("markdown", None)

        if md is None:
            continue  # nothing to save

        md_path = out_dir / f"{cid}.md"
        if md_path.exists():
            continue

        # md might be a list of lines or a single string depending on the dataset
        if isinstance(md, list):
            text = "\n".join(md)
        else:
            text = str(md)

        with open(md_path, "w") as f:
            f.write(text)


def run_docling():
    converter = DocumentConverter()

    src_root = Path(os.getcwd())
    for img_dir in [d for d in src_root.rglob("*/images") if d.is_dir()]:
        images = natsorted([p for p in img_dir.glob("*")
                            if p.is_file()], alg=ns.IGNORECASE)

        out_dir = img_dir.parent / "texts"
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in tqdm(images, desc=f"Converting in {img_dir}", total=len(images)):
            out_file = out_dir / f"{img.stem}.md"
            if out_file.exists():
                continue
            img = Image.open(img)
            img = safe_resize(img)

            # -------- robust conversion with up‑to‑3 automatic retries ------
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    tmp_path = pil_image_to_temp_path(img)
                    result = converter.convert(tmp_path)
                    break  # success
                except cv2.error as e:
                    print("------------------")
                    print(out_file)
                    if "SHRT_MAX" not in str(e) or attempt == MAX_RETRIES:
                        raise  # different error or out of retries
                    # shrink more and retry
                    w, h = img.size
                    img = img.resize((int(w * 0.8), int(h * 0.8)),
                                     DOWNSAMPLE_FILTER)

            with open(out_file, "w") as md_out:
                md_out.write(result.document.export_to_markdown())


if __name__ == "__main__":
    from dataset_configs import VIDORE1_DATASETS, VIDORE2_DATASETS, VIDORE3_DATASETS
    for dataset in [*VIDORE3_DATASETS]:
        dataset_name, subset_name = dataset.split("/")
        dataset_hf_id = f"vidore/{subset_name}"
        # download_hf_queries(dataset_hf_id, out_dir=Path(dataset_name) / subset_name / "benchmark")
        # download_hf_images(dataset_hf_id, "corpus", out_dir=Path(dataset_name)/ subset_name / "images")
        download_hf_markdown(dataset_hf_id,"corpus",out_dir=Path(dataset_name) / subset_name / "texts")

    # run_docling()
