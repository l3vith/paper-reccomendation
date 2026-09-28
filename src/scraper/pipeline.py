"""
Orchestrator module for the scraping pipeline.

This module combines arXiv and Semantic Scholar scrapers, manages data enrichment,
stores results in the database, and exports them to a JSONL file.
"""

import os
import logging
import argparse
from typing import List

from .arxiv_scraper import search_arxiv
from .semantic_scholar_scraper import search_semantic_scholar, enrich_with_citations
from .firecrawl_scraper import search_firecrawl
from .data_store import PaperStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(topics: List[str], max_per_topic: int = 500, enrich_citations: bool = True, max_citation_enrichment: int = 100, db_path: str = 'data/papers.db', use_firecrawl: bool = True, firecrawl_max_per_topic: int = 50) -> None:
    """
    Run the complete scraping pipeline.

    Args:
        topics (List[str]): List of topics/queries to search for.
        max_per_topic (int): Maximum number of papers to fetch per topic per source.
        enrich_citations (bool): Whether to enrich Semantic Scholar papers with citations.
        max_citation_enrichment (int): Maximum number of papers to enrich per topic.
        db_path (str): Database file path.
    """
    store = PaperStore(db_path=db_path)
    
    total_arxiv = 0
    total_s2 = 0
    total_fc = 0
    
    for topic in topics:
        logger.info(f"=== Starting pipeline for topic: '{topic}' ===")
        
        # 1. Scrape arXiv
        logger.info("Scraping arXiv...")
        arxiv_papers = search_arxiv(topic, max_results=max_per_topic)
        total_arxiv += len(arxiv_papers)
        total_fc += len(fc_papers)
        
        # 2. Scrape Semantic Scholar
        logger.info("Scraping Semantic Scholar...")
        s2_papers = search_semantic_scholar(topic, max_results=max_per_topic)
        total_s2 += len(s2_papers)
        
        # 3. Enrich Semantic Scholar papers if enabled
        if enrich_citations and s2_papers:
            logger.info("Enriching Semantic Scholar papers with citations...")
            s2_papers = enrich_with_citations(s2_papers, max_papers=max_citation_enrichment)

        # 3b. Search Firecrawl research index (titles + abstracts + PDF links)
        fc_papers = []
        if use_firecrawl:
            logger.info("Searching Firecrawl research index...")
            try:
                fc_papers = search_firecrawl(topic, max_results=firecrawl_max_per_topic)
            except Exception as exc:
                logger.warning(f"Firecrawl source skipped: {exc}")

        # 4. Merge and store
        logger.info("Storing papers in database...")
        inserted_arxiv = store.insert_papers(arxiv_papers)
        inserted_s2 = store.insert_papers(s2_papers)
        inserted_fc = store.insert_papers(fc_papers) if fc_papers else 0

        logger.info(f"Topic '{topic}' summary: Inserted {inserted_arxiv}/{len(arxiv_papers)} arXiv, {inserted_s2}/{len(s2_papers)} Semantic Scholar and {inserted_fc}/{len(fc_papers)} Firecrawl papers (skipped duplicates).")

    # Final Summary and Export
    total_db_count = store.count()
    logger.info(f"=== Pipeline Complete ===")
    logger.info(f"Total papers scraped: arXiv={total_arxiv}, Semantic Scholar={total_s2}, Firecrawl={total_fc}")
    logger.info(f"Total papers in database: {total_db_count}")
    
    export_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), 'papers.jsonl')
    logger.info(f"Exporting all data to {export_path}...")
    store.export_jsonl(output_path=export_path)
    logger.info("Done.")

def main():
    parser = argparse.ArgumentParser(description="Research Paper Recommendation System - Data Collection Pipeline")
    parser.add_argument("--topics", type=str, required=True, help="Comma-separated list of topics (e.g., 'NLP,transformers,attention mechanism')")
    parser.add_argument("--max-per-topic", type=int, default=500, help="Maximum number of results to fetch per topic per source")
    parser.add_argument("--no-citations", action="store_true", help="Skip citation enrichment for Semantic Scholar papers")
    parser.add_argument("--no-firecrawl", action="store_true", help="Skip the Firecrawl research-index source")
    parser.add_argument("--firecrawl-max", type=int, default=50, help="Maximum Firecrawl papers per topic")
    parser.add_argument("--db-path", type=str, default="data/papers.db", help="Path to the SQLite database file")
    
    args = parser.parse_args()
    
    topic_list = [t.strip() for t in args.topics.split(",") if t.strip()]
    
    run_pipeline(
        topics=topic_list,
        max_per_topic=args.max_per_topic,
        enrich_citations=not args.no_citations,
        max_citation_enrichment=100,
        use_firecrawl=not args.no_firecrawl,
        firecrawl_max_per_topic=args.firecrawl_max,
        db_path=args.db_path
    )

if __name__ == "__main__":
    main()
