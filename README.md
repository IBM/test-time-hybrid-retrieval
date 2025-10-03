# Multimodal Hybrid Retrieval with Guided Query Refinement (GQR)
This repository provides an experimental framework for testing visual document retrieval methods, focusing on hybrid retrieval that combines vision-centric multimodal encoders and semantic text encoders. 

It contains an implementation of a novel hybrid method - **Guided Query Refinement (GQR)** - as introduced in *Uzan et al. 2025*.

GQR is a test-time optimization approach that refines the query representation of a *primary* retriever using guidance from the query-documents similarity scores of a *complementary* retriever.

## 📂 Project Structure
The repository is organized into several key components:

**Execution Scripts**:

`ingest_datasets.py`: Downloads and preprocess datasets from the ViDoRe1 and ViDoRe2 benchmarks.

`calc_embeddings.py`: Pre-computes and saves embeddings for specified models and datasets.

`run_experiments.py`: The main script to run hybrid retrieval and query optimization experiments.

`bench_latency_vidore2.py`: Measures the latency of various retrieval and fusion methods.

`run_reranker.py`: Runs an experiment using a cross-encoder reranker, on top of a base retriever.

**Core Logic**:


`retriever.py`: Defines the Retriever class that manages loading embeddings and computing scores.

`query_optimizations.py`: Contains the implementation of the query optimization algorithms.

`fusion_methods.py`: Implements basic late fusion algorithms, including RRF and score aggregation methods .

**Configuration**:

`dataset_configs.py`: Defines the relevant benchmark datasets, as well as the data loading and evaluation logic.

`embedding_configs.py`: Configures all supported embedding models, and their model loading and embedding calculation code.

## ⚙️ Setup and Installation
1. Clone the repository:
```bash
 git clone https://github.com/IBM/test-time-hybrid-retrieval.git
 cd test-time-hybrid-retrieval
 ```

2. Create and activate a virtual environment (recommended)

3. Install the project requirements:

```
 pip install -r requirements.txt
```

4. For optimal performance on supported hardware, it is highly recommended to install flash-attention:

```
 pip install flash-attn 
```

## ▶️ Usage and Workflow
Follow these steps to replicate the experiments.
### Step 1: Ingest and Process Datasets
First, download the required datasets from the Hugging Face Hub. The script uses [docling](https://github.com/docling-project/docling) to convert document images into markdown text for text-based encoders.

`python ingest_datasets.py` - This will create directories (e.g., Vidore1/, Vidore2/) containing the datasets, benchmarks, images, and ingested texts.

### Step 2: Pre-compute Embeddings
Next, pre-compute the document and query embeddings for all models you wish to experiment with.
Specify the models using the `--models` argument and the datasets with `—datasets`.

Example for calculating embeddings for multiple models on Vidore2 datasets:

```bash
python calc_embeddings.py \
    --models nvidia colnomic jina_multi qwen_text linq \
    --datasets Vidore2/esg_reports_v2 Vidore2/biomedical_lectures_v2 Vidore2/economics_reports_v2 Vidore2/esg_reports_human_labeled_v2
```

This will create *_embeddings subdirectories inside each dataset folder.

**Note**: Some models that were used in the paper require specific versions of some packages in order to calculate embeddings. For the most up-to-date information, it is best to check the HF page for each model before using it: [Llama-Nemo](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1), [ColNomic](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b), [Jina-v4](https://huggingface.co/jinaai/jina-embeddings-v4), [Linq-Embed](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

### Step 3: Run Retrieval Experiments
`run_experiments.py` is the main script to evaluate the hybrid retrieval and query optimization methods. The args for the script are:

`--models`: Specify which models to run (`nvidia`, `jina_multi`, `colnomic`, `linq`, `qwen_text`, `jina_text_multi`)

`--benchmarks`: Specify which benchmark sets to run on (`vidore1`, `vidore2`).

`--hyper_config`: Path to the config file with hyperparameters grid for tuning.

`--datasets_path_prefix`: Path to the datasets directory on your machine.

`--tune`: Set to True to perform hyperparameter tuning on a dev split. If False, it runs all combinations.

`-p`, `--use_parallelization`: Set to True to parallelize experiments across CPU cores.

`-o`, `--out_dir_suffix`: An optional suffix for the output directory name.

**Example:** Run experiments on Vidore2 with hyperparameter tuning:
```bash
python run_experiments.py \
    --models colnomic linq \
    --benchmarks vidore2 \
    --tune True \
    -o "vidore2_tuned_run"
```

### 📊 Results
The output of the experiments will be saved in the output/ directory.

`run_experiments.py` saves results to output/results-<hash><suffix>/. 

Inside, you will find CSV files for each metric (e.g., ndcg@5.csv, recall@10.csv) summarizing the performance across all datasets.
`run_reranker.py` and `bench_latency_vidore2.py` save their results in similarly structured output directories, containing performance metrics and latency measurements in JSON format.


### 📜 Citation

Coming soon!


