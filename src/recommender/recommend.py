"""
High-level recommendation logic.
"""

import logging
from typing import Any, Dict, List, Optional

from src.recommender.indexer import PaperIndex
from src.scraper.data_store import PaperStore
from src.model.query_expansion import ScientificQueryExpander

# Import scrapers if they exist
try:
    from src.scraper import arxiv_scraper, semantic_scholar_scraper, firecrawl_scraper
except ImportError:
    arxiv_scraper = None
    semantic_scholar_scraper = None
    firecrawl_scraper = None

logger = logging.getLogger(__name__)


def _diversify_sources(results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Interleave source-ranked candidates so one provider cannot hide another."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        source = str(result.get("source") or "unknown").lower()
        buckets.setdefault(source, []).append(result)
    if len(buckets) < 2:
        return results[:top_k]

    # Keep each provider's score ordering, start with whichever provider has
    # the strongest candidate, then alternate while both have candidates.
    source_order = sorted(
        buckets,
        key=lambda source: float(buckets[source][0].get("score", 0.0)),
        reverse=True,
    )
    diversified = []
    while len(diversified) < top_k and any(buckets.values()):
        for source in source_order:
            if buckets[source] and len(diversified) < top_k:
                diversified.append(buckets[source].pop(0))
    return diversified


class PaperRecommender:
    """Provides high-level paper recommendation capabilities with query expansion."""

    def __init__(
        self,
        model_path: str = "models/scibert-finetuned-papers",
        index_path: str = "data/faiss_index.bin",
        mapping_path: str = "data/paper_id_map.json",
        pooling: str = "mean",
        db_path: str = "data/papers.db",
        document_format: str = "sep",
        max_seq_length: int | None = None,
        require_checkpoint: bool = False,
    ) -> None:
        """
        Initialize the recommender.

        Args:
            model_path: Path to the embedding model.
            index_path: Path to the FAISS index.
            mapping_path: Path to the ID mapping.
            db_path: Path to the local database.
        """
        self.index = PaperIndex(model_path, index_path, mapping_path, pooling=pooling,
                                document_format=document_format, max_seq_length=max_seq_length,
                                require_checkpoint=require_checkpoint)
        self.store = PaperStore(db_path)
        self.expander = ScientificQueryExpander(db_path=db_path)
        self.last_search_diagnostics: Dict[str, Any] = {}

    def recommend_by_paper(self, title: str, abstract: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend similar papers based on a given paper's title and abstract.

        Args:
            title: Title of the reference paper.
            abstract: Abstract of the reference paper.
            top_k: Number of recommendations.

        Returns:
            List of recommended papers.
        """
        query_text = self.index.paper_text(title, abstract)
        return self.index.search(query_text, top_k=top_k)

    def recommend_by_topic(self, topic: str, top_k: int = 10, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend papers related to a given topic with optional scientific query expansion.

        Args:
            topic: Topic string to search for.
            top_k: Number of recommendations.
            use_expansion: Whether to expand query with scientific domain terms and acronyms.

        Returns:
            List of recommended papers.
        """
        clean_topic = topic
        if arxiv_scraper and hasattr(arxiv_scraper, "clean_and_normalize_query"):
            clean_topic, _ = arxiv_scraper.clean_and_normalize_query(topic)

        if use_expansion:
            expansion = self.expander.expand(clean_topic)
            search_query = expansion["dense_query"]
            logger.info(f"Expanded query: '{clean_topic}' -> '{search_query}'")
        else:
            search_query = clean_topic

        return self.index.search(search_query, top_k=top_k)

    def recommend_adaptive(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.40,
        min_good_results: int = 3,
        auto_train: bool = True,
        use_lora: bool = True,
        max_scrape: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Intelligently recommend papers using query expansion and sparsity detection.

        Args:
            query: The search topic or query text.
            top_k: Number of recommendations to return.
            similarity_threshold: Cosine similarity threshold below which data is considered sparse.
            min_good_results: Minimum number of papers with score >= threshold required.
            auto_train: Whether to trigger model fine-tuning if new papers are added.
            use_lora: Whether to use LoRA parameter-efficient adaptation during fine-tuning.
            max_scrape: Max papers to scrape if sparsity is detected.

        Returns:
            List of recommended papers.
        """
        clean_query = query
        if arxiv_scraper and hasattr(arxiv_scraper, "clean_and_normalize_query"):
            clean_query, _ = arxiv_scraper.clean_and_normalize_query(query)
            
        expansion = self.expander.expand(clean_query)
        dense_query = expansion["dense_query"]
        scraper_query = expansion["scraper_query"]

        # Step 1: Query local index with expanded dense query
        results = self.index.search(dense_query, top_k=top_k)
        
        # Sparsity is about semantic cosine similarity, not the lexical rerank.
        good_matches = [r for r in results if r.get('dense_score', r.get('score', 0)) >= similarity_threshold]
        is_sparse = len(good_matches) < min_good_results or (results and results[0].get('dense_score', results[0].get('score', 0)) < similarity_threshold) or len(results) == 0

        # Always refresh arXiv for topic searches. Returning early on a good
        # local cosine score used to prevent arXiv from being queried at all,
        # leaving the UI dominated by whichever provider populated the index
        # first. Semantic Scholar remains the broader fallback for sparse topics.
        top_score = results[0].get('dense_score', results[0]['score']) if results else 0.0
        logger.info("Refreshing arXiv candidates for '%s' (local top cosine %.4f).", clean_query, top_score)

        new_papers = []
        self.last_search_diagnostics = {
            "arxiv": {"count": 0, "errors": [], "rate_limited": False, "request_rejected": False}
        }
        if arxiv_scraper:
            logger.info(f"Live scraping arXiv for query: '{clean_query}'...")
            try:
                # Pass the untouched user query so the scraper's broad
                # all-field fallback preserves terms such as "diseases".
                arxiv_papers = arxiv_scraper.search_arxiv(query, max_results=max_scrape)
                new_papers.extend(arxiv_papers)
                self.last_search_diagnostics["arxiv"] = dict(
                    getattr(arxiv_scraper, "last_search_status", {})
                )
            except Exception as e:
                self.last_search_diagnostics["arxiv"] = {
                    "count": 0, "errors": [f"{type(e).__name__}: {e}"],
                    "rate_limited": "429" in str(e),
                    "request_rejected": "406" in str(e),
                }
                logger.error(f"Error scraping arXiv: {e}")

        if is_sparse and semantic_scholar_scraper:
            logger.info(f"Live scraping Semantic Scholar for query: '{clean_query}'...")
            try:
                ss_papers = semantic_scholar_scraper.search_semantic_scholar(clean_query, max_results=max_scrape)
                new_papers.extend(ss_papers)
            except Exception as e:
                logger.error(f"Error scraping Semantic Scholar: {e}")

        # Firecrawl research index: independent third source, and the fallback
        # when arXiv rejects the request (HTTP 406) or returns nothing.
        arxiv_failed = not new_papers or self.last_search_diagnostics.get("arxiv", {}).get("request_rejected", False)
        if firecrawl_scraper and (is_sparse or arxiv_failed):
            logger.info(f"Live searching Firecrawl research index for query: '{clean_query}'...")
            try:
                fc_papers = firecrawl_scraper.search_firecrawl(clean_query, max_results=min(max_scrape, 40))
                new_papers.extend(fc_papers)
            except Exception as e:
                logger.error(f"Error searching Firecrawl: {e}")

        if not new_papers:
            logger.warning("No new online papers found for '%s'; serving local matches.", clean_query)
            if not is_sparse:
                logger.info("Sufficient local matches found; no Semantic Scholar fetch was needed.")
            return results

        # Save new papers to SQLite store
        inserted_count = self.store.insert_papers(new_papers)
        logger.info(f"Stored {inserted_count} newly scraped papers in database.")

        # Step 3: Re-index newly fetched papers into FAISS (instant embedding)
        logger.info(f"Re-indexing {len(new_papers)} papers into FAISS...")
        self.index.add_papers(new_papers)

        # Step 4: Conditionally update triplets dataset for offline fine-tuning
        if auto_train and inserted_count > 0:
            try:
                from src.model.build_triplets import load_papers, generate_triplets, save_triplets
                all_papers = load_papers(self.store.db_path)
                triplets = generate_triplets(all_papers, max_positives_per_anchor=2, hard_negatives=True)
                if triplets:
                    save_triplets(triplets, "data/triplets.csv")
                    logger.info(f"Updated training triplets ({len(triplets)} pairs) for fine-tuning.")
            except Exception as e:
                logger.debug(f"Triplets export skipped: {e}")

        # Step 5: Return search results from updated FAISS index
        # Search a wider pool before source interleaving; otherwise an arXiv
        # result ranked just below top_k could never make it into the display.
        updated_results = self.index.search(dense_query, top_k=max(top_k * 5, 50))
        diversified = _diversify_sources(updated_results, top_k)
        source_counts: Dict[str, int] = {}
        for result in diversified:
            source = str(result.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.info("Returning %d recommendations with source mix: %s", len(diversified), source_counts)
        return diversified

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format the search results as a nice text table.

        Args:
            results: List of result dictionaries from search.

        Returns:
            Formatted string.
        """
        if not results:
            return "No results found."
            
        header = f"{'Rank':<4} | {'Score':<6} | {'Title':<30} | {'Year':<4} | {'URL'}"
        separator = f"{'-'*4}-+-{'-'*6}-+-{'-'*30}-+-{'-'*4}-+-{'-'*27}"
        
        lines = [header, separator]
        
        for res in results:
            rank = res.get('rank', '?')
            score = f"{res.get('score', 0):.4f}"
            title = res.get('title', '')
            if len(title) > 30:
                title = title[:27] + '...'
            year = str(res.get('year', 'N/A'))
            url = res.get('pdf_url') or res.get('external_id') or 'N/A'
            
            line = f"{rank:<4} | {score:<6} | {title:<30} | {year:<4} | {url}"
            lines.append(line)
            
        return "\n".join(lines)
