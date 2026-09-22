# Research Paper Recommendation System

A complete end-to-end research paper recommendation system that aggregates papers from arXiv and Semantic Scholar, fine-tunes a SciBERT model for better document embeddings, and provides fast recommendations using a FAISS vector index. 

This system allows you to build a local knowledge base of research papers and query it effectively using natural language queries, or by finding papers similar to an existing one.

## Architecture

The system consists of three main phases:
1. **Scraping**: Connects to arXiv and Semantic Scholar APIs to download metadata and abstracts.
2. **Modeling**: Generates training triplets (anchor, positive, negative) and fine-tunes a sentence-transformers model (SciBERT base).
3. **Recommendation Engine**: Builds a FAISS (Facebook AI Similarity Search) index for high-speed vector similarity search.

## Installation

Using **uv** (recommended):
```bash
# Clone the repository
git clone git@github.com:l3vith/paper-reccomendation.git
cd paper-reccomendation

# Sync environment and install dependencies
uv sync
```

Alternatively using standard pip:
```bash
pip install -r requirements.txt
```

> **Note**: GPU is helpful but optional. The project runs on CUDA when available, Apple Silicon MPS on macOS, and CPU on Windows or Linux. On a CPU-only laptop, choose batch size 2 or 4 and expect fine-tuning to take longer.

### Windows

The app supports Windows 10/11 with Python 3.11+. Install [uv](https://docs.astral.sh/uv/) or use pip, then run the same commands above from PowerShell:

```powershell
uv sync
uv run streamlit run app_streamlit.py
```

Text-to-speech uses the installed Windows SAPI voice and produces a standard PCM WAV file for browser playback. No Apple-specific audio tools are required on Windows.

## Web Application (Streamlit UI)

Launch the interactive web frontend:
```bash
uv run streamlit run app_streamlit.py
```
This opens the web interface with:
- **Adaptive Discovery**: Local FAISS search with automatic live web scraping and model updating on sparse topics.
- **Seed Paper Similarity**: Input reference paper Title and Abstract.
- **Synchronized PDF Reader**: View the downloaded PDF in-browser, read the current page aloud, and follow the highlighted word in a live transcript.
- **Four-model benchmark dashboard**: Fine-tune and compare four controlled SciBERT variants in one held-out-metrics table.
- **Corpus Explorer**: Searchable table of all stored research papers.

## Quick Start (with `uv`)

### Option A: One-Command Automated Pipeline
Run everything automatically (Scrape → Triplets → Fine-Tune → Benchmark Eval → Index → Recommend):
```bash
uv run python run_pipeline.py --topics "NLP,transformers" --max-per-topic 100 --query "attention mechanism"
```
> **Note**: The script automatically skips fine-tuning if the model has already been trained once (use `--force-retrain` if you wish to retrain).

### Option B: Step-by-Step Execution
```bash
# 1. Scrape papers
uv run python -m src.scraper.pipeline --topics "NLP,transformers" --max-per-topic 500

# 2. Build triplets for training
uv run python -m src.model.build_triplets

# 3. Train the embedding model (one-time step)
uv run python -m src.model.train --epochs 4

# 4. Build FAISS Index & Run recommendations
uv run python -m src.recommender.app index
uv run python -m src.recommender.app recommend --mode topic --query "attention mechanism" --top-k 10
```

## Project Structure

```
course-project/
├── data/               # SQLite db, FAISS index, paper ID maps
├── models/             # Fine-tuned embedding models
├── src/
│   ├── scraper/        # API integrations for arXiv & Semantic Scholar
│   ├── model/          # Triplet generation, fine-tuning & query expansion
│   │   ├── build_triplets.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── query_expansion.py  # Scientific query expansion module
│   └── recommender/    # FAISS indexer and CLI application
├── app_streamlit.py    # Streamlit web frontend
├── run_pipeline.py     # Master orchestrator
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Phases

- **Phase 1: Data Collection (`src/scraper`)**: Aggregates metadata into a unified SQLite format (`data/papers.db`).
- **Phase 2: Representation Learning (`src/model`)**: Creates semantic embeddings that accurately capture technical domain similarities.
- **Phase 3: Retrieval (`src/recommender`)**: Employs Inner Product vector search to retrieve the most relevant papers instantly.

## Four SciBERT variants for the study

The sidebar selects the model used for recommendations. Every variant retains
the same SciBERT encoder and TripletLoss setup; only its pooling architecture
changes. This makes the comparison reproducible while producing distinct paper
representations.

| Variant | Pooling head | Embedding size |
|---|---|---:|
| SciBERT Mean Pooling | Original mean token pooling | 768 |
| SciBERT [CLS] Pooling | Final `[CLS]` token | 768 |
| SciBERT Mean + Max Pooling | Concatenated mean and max pooling | 1,536 |
| SciBERT Weighted-Mean Pooling | Position-weighted mean pooling | 768 |

Use **Model Benchmarks → Fine-tune selected model** for each profile, then
select **Evaluate all trained variants**. The resulting table reports actual
Triplet Accuracy, MRR, Precision@5, and Precision@10 on the same holdout.
Each variant owns a separate FAISS index, so every active search uses the
matching model embeddings.
