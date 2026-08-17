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
git clone https://github.com/yourusername/course-project.git
cd course-project

# Sync environment and install dependencies
uv sync
```

Alternatively using standard pip:
```bash
pip install -r requirements.txt
```

> **Note**: GPU is highly recommended for the model fine-tuning and inference steps. For CPU-only environments, adjust batch sizes and expect longer processing times.

## Web Application (Streamlit UI)

Launch the interactive web frontend:
```bash
uv run streamlit run app_streamlit.py
```
This opens the web interface with:
- **Adaptive Discovery**: Local FAISS search with automatic live web scraping and model updating on sparse topics.
- **Seed Paper Similarity**: Input reference paper Title and Abstract.
- **Benchmark Dashboard**: Quantitative comparison of Fine-Tuned SciBERT vs. Base SciBERT.
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
