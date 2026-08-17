#!/usr/bin/env python3
"""
Master Pipeline Orchestrator for Research Paper Recommendation System.

This script allows you to run the complete end-to-end workflow or individual stages:
  1. Scrape papers from arXiv & Semantic Scholar
  2. Generate citation-based training triplets
  3. Fine-tune SciBERT model (ONE-TIME step, skipped if already trained)
  4. Evaluate fine-tuned model vs base model
  5. Build / Update FAISS vector index
  6. Query recommendations (topic, paper, or live)
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("Pipeline")


def run_scraping(topics: list[str], max_per_topic: int = 100, db_path: str = "data/papers.db"):
    logger.info(f"===> [Phase 1] Scraping papers for topics: {topics}")
    from src.scraper.pipeline import run_pipeline as scrape_pipeline
    scrape_pipeline(
        topics=topics,
        max_per_topic=max_per_topic,
        enrich_citations=True,
        max_citation_enrichment=50,
        db_path=db_path
    )


def run_triplets(db_path: str = "data/papers.db", output_path: str = "data/triplets.csv", hard_negatives: bool = True):
    logger.info("===> [Phase 2.1] Generating training triplets from citation graph...")
    from src.model.build_triplets import load_papers, generate_triplets, save_triplets
    papers = load_papers(db_path)
    if not papers:
        logger.error(f"No papers found in database at {db_path}. Please run scraping first.")
        sys.exit(1)
    
    triplets = generate_triplets(papers, max_positives_per_anchor=5, hard_negatives=hard_negatives)
    if not triplets:
        logger.warning("No citation triplets could be formed. Using non-citation pairs or scraping more papers is recommended.")
    save_triplets(triplets, output_path)


def run_training(triplets_path: str = "data/triplets.csv",
                 output_model_path: str = "models/scibert-finetuned-papers",
                 epochs: int = 3,
                 batch_size: int = 16,
                 use_lora: bool = True,
                 force_retrain: bool = False):
    logger.info("===> [Phase 2.2] Fine-Tuning Model with LoRA...")
    if os.path.exists(output_model_path) and not force_retrain:
        logger.info(f"Model already exists at '{output_model_path}'. Skipping training (use --force-retrain to override).")
        return

    from src.model.train import load_training_data, create_model, train
    if not os.path.exists(triplets_path):
        logger.error(f"Triplets file '{triplets_path}' not found. Run triplet generation first.")
        sys.exit(1)

    train_data, _ = load_training_data(triplets_path)
    if len(train_data) == 0:
        logger.error("Triplets dataset is empty. Cannot train.")
        sys.exit(1)

    model = create_model("allenai/scibert_scivocab_uncased", use_lora=use_lora)
    train(
        model=model,
        train_examples=train_data,
        output_path=output_model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5
    )


def run_evaluation(model_path: str = "models/scibert-finetuned-papers",
                   triplets_path: str = "data/triplets.csv",
                   db_path: str = "data/papers.db"):
    logger.info("===> [Phase 2.3] Evaluating Model...")
    from src.model.evaluate import evaluate_and_compare
    if not os.path.exists(model_path):
        logger.warning(f"Fine-tuned model at '{model_path}' not found. Skipping evaluation.")
        return

    results = evaluate_and_compare(
        finetuned_path=model_path,
        base_model_name="allenai/scibert_scivocab_uncased",
        triplets_path=triplets_path,
        db_path=db_path
    )
    print("\n" + "=" * 55)
    print("               EVALUATION BENCHMARK RESULTS")
    print("=" * 55)
    print(f"{'Metric':<22} | {'Base SciBERT':<14} | {'Fine-Tuned'}")
    print("-" * 55)
    for m in ['triplet_accuracy', 'mrr', 'precision@5', 'precision@10']:
        base_v = results.get("Base Model", {}).get(m, 0.0)
        ft_v = results.get("Fine-tuned Model", {}).get(m, 0.0)
        print(f"{m:<22} | {base_v:<14.4f} | {ft_v:.4f}")
    print("=" * 55 + "\n")


def run_indexing(db_path: str = "data/papers.db", model_path: str = "models/scibert-finetuned-papers"):
    logger.info("===> [Phase 3.1] Building FAISS Vector Index...")
    from src.recommender.indexer import PaperIndex
    indexer = PaperIndex(model_path=model_path)
    count = indexer.build_index(db_path=db_path)
    logger.info(f"Successfully indexed {count} papers into FAISS.")


def run_recommend(query: str, top_k: int = 5, mode: str = "adaptive", threshold: float = 0.40, auto_train: bool = True, model_path: str = "models/scibert-finetuned-papers"):
    logger.info(f"===> [Phase 3.2] Running Recommendation (mode={mode}, query='{query}')...")
    from src.recommender.recommend import PaperRecommender
    recommender = PaperRecommender(model_path=model_path)
    
    if mode == "adaptive":
        results = recommender.recommend_adaptive(
            query=query,
            top_k=top_k,
            similarity_threshold=threshold,
            auto_train=auto_train,
            max_scrape=50
        )
    elif mode == "live":
        results = recommender.recommend_adaptive(
            query=query,
            top_k=top_k,
            similarity_threshold=0.99,
            auto_train=auto_train,
            max_scrape=50
        )
    else:
        results = recommender.recommend_by_topic(query, top_k=top_k)

    print("\n" + "=" * 80)
    print(f"RECOMMENDED PAPERS FOR: '{query}'")
    print("=" * 80)
    print(recommender.format_results(results))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline for Research Paper Recommender")
    
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run entire pipeline from scratch (Scrape -> Triplets -> Train -> Eval -> Index -> Recommend)")
    parser.add_argument("--topics", type=str, default="NLP,transformer models,attention mechanism",
                        help="Comma-separated topics to scrape")
    parser.add_argument("--max-per-topic", type=int, default=100,
                        help="Max papers to scrape per topic per source")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs for SciBERT fine-tuning")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force fine-tuning even if model already exists")
    parser.add_argument("--query", type=str, default="attention mechanism in transformers",
                        help="Sample query for recommendation")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of recommendations to return")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Cosine similarity threshold below which data is considered sparse")
    parser.add_argument("--recommend-only", action="store_true",
                        help="Only run recommendation (adaptively fetches + re-indexes + trains ONLY if sparse)")

    args = parser.parse_args()

    topic_list = [t.strip() for t in args.topics.split(",") if t.strip()]

    # Case 1: Recommendation only
    if args.recommend_only:
        run_recommend(query=args.query, top_k=args.top_k)
        return

    # Case 2: Full pipeline or default execution
    logger.info("Starting Research Paper Recommendation Pipeline...")
    
    # 1. Scrape
    run_scraping(topics=topic_list, max_per_topic=args.max_per_topic)
    
    # 2. Triplets
    run_triplets()
    
    # 3. Fine-Tune (skipped if already fine-tuned, unless --force-retrain is given)
    run_training(epochs=args.epochs, force_retrain=args.force_retrain)
    
    # 4. Evaluate
    run_evaluation()
    
    # 5. Index
    run_indexing()
    
    # 6. Recommendation
    run_recommend(query=args.query, top_k=args.top_k)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
