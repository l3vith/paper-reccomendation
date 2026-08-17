"""
CLI entry point for the recommendation system.
"""

import argparse
import logging
import sys

from src.recommender.recommend import PaperRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Research Paper Recommendation System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index command
    index_parser = subparsers.add_parser("index", help="Build/rebuild the FAISS index")
    index_parser.add_argument("--db-path", default="data/papers.db", help="Path to SQLite database")
    index_parser.add_argument("--model-path", default="models/scibert-finetuned-papers", help="Path to embedding model")

    # recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Recommend papers")
    recommend_parser.add_argument("--mode", choices=["paper", "topic", "live", "adaptive"], default="adaptive", help="Recommendation mode (default: adaptive)")
    recommend_parser.add_argument("--query", help="Query topic (for topic, live, or adaptive mode)")
    recommend_parser.add_argument("--title", help="Paper title (for paper mode)")
    recommend_parser.add_argument("--abstract", help="Paper abstract (for paper mode)")
    recommend_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    recommend_parser.add_argument("--threshold", type=float, default=0.40, help="Similarity threshold for sparsity detection (adaptive mode)")
    recommend_parser.add_argument("--max-scrape", type=int, default=50, help="Max papers to scrape if sparse")
    recommend_parser.add_argument("--no-train", action="store_true", help="Disable auto-training when sparse data is found")

    # scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape papers")
    scrape_parser.add_argument("--topics", required=True, help="Comma-separated topics")
    scrape_parser.add_argument("--max-per-topic", type=int, default=500, help="Max papers per topic")

    # full-pipeline command
    pipeline_parser = subparsers.add_parser("full-pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("--topics", required=True, help="Topics to scrape")
    pipeline_parser.add_argument("--query", required=True, help="Query to recommend")
    pipeline_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "index":
            from src.recommender.indexer import PaperIndex
            indexer = PaperIndex(model_path=args.model_path)
            count = indexer.build_index(db_path=args.db_path)
            logger.info(f"Successfully indexed {count} papers.")

        elif args.command == "recommend":
            recommender = PaperRecommender()
            results = []
            
            if args.mode == "paper":
                if not args.title or not args.abstract:
                    logger.error("Paper mode requires --title and --abstract")
                    sys.exit(1)
                results = recommender.recommend_by_paper(args.title, args.abstract, args.top_k)
            
            elif args.mode == "topic":
                if not args.query:
                    logger.error("Topic mode requires --query")
                    sys.exit(1)
                results = recommender.recommend_by_topic(args.query, args.top_k)
                
            elif args.mode == "adaptive":
                if not args.query:
                    logger.error("Adaptive mode requires --query")
                    sys.exit(1)
                results = recommender.recommend_adaptive(
                    args.query,
                    top_k=args.top_k,
                    similarity_threshold=args.threshold,
                    auto_train=not args.no_train,
                    max_scrape=args.max_scrape
                )
                
            elif args.mode == "live":
                if not args.query:
                    logger.error("Live mode requires --query")
                    sys.exit(1)
                results = recommender.recommend_adaptive(
                    args.query,
                    top_k=args.top_k,
                    similarity_threshold=0.99,  # Force live scrape
                    auto_train=not args.no_train,
                    max_scrape=args.max_scrape
                )
                
            print("\nRecommendations:\n")
            print(recommender.format_results(results))

        elif args.command == "scrape":
            from src.scraper.pipeline import run_pipeline
            topics = [t.strip() for t in args.topics.split(",")]
            run_pipeline(topics, max_per_topic=args.max_per_topic)
            logger.info("Scraping completed.")

        elif args.command == "full-pipeline":
            # 1. Scrape
            from src.scraper.pipeline import run_pipeline
            topics = [t.strip() for t in args.topics.split(",")]
            logger.info("Starting scraping...")
            run_pipeline(topics, max_per_topic=100)
                
            # 2. Index
            from src.recommender.indexer import PaperIndex
            logger.info("Building index...")
            indexer = PaperIndex()
            indexer.build_index()
            
            # 3. Recommend
            logger.info("Generating recommendations...")
            recommender = PaperRecommender()
            results = recommender.recommend_by_topic(args.query, args.top_k)
            print("\nRecommendations:\n")
            print(recommender.format_results(results))
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
