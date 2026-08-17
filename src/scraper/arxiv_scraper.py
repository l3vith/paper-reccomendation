"""
Module for scraping papers from arXiv.

This module provides functions to search arXiv for research papers by topic or keyword,
extract relevant metadata, and return the results as a list of dictionaries.
"""

import logging
import arxiv
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import difflib

# Core Scientific & Academic Vocabulary for Fuzzy Spell-Correction
SCIENTIFIC_VOCABULARY = [
    "protein", "proteins", "structure", "structural", "function", "prediction",
    "transformer", "transformers", "attention", "mechanism", "diffusion", "latent",
    "generative", "recommendation", "recommender", "retrieval", "augmented", "generation",
    "embedding", "embeddings", "encoder", "decoder", "representation", "learning",
    "classification", "convolution", "convolutional", "recurrent", "network", "networks",
    "graph", "neural", "reinforcement", "optimization", "quantum", "approximate",
    "molecular", "biology", "genomics", "sequence", "alignment", "biomedical",
    "language", "models", "supervised", "unsupervised", "multimodal", "synthetic"
]

def clean_and_normalize_query(raw_query: str) -> tuple[str, str]:
    """
    Correct typos using fuzzy matching and build high-precision arXiv search query syntax.
    
    Returns:
        (clean_text, structured_arxiv_query)
    """
    tokens = raw_query.strip().split()
    clean_tokens = []
    
    for t in tokens:
        t_clean = t.strip().lower()
        if not t_clean:
            continue
            
        # 1. Check exact match in vocabulary
        if t_clean in SCIENTIFIC_VOCABULARY:
            clean_tokens.append(t_clean)
        else:
            # 2. Fuzzy match against vocabulary (similarity >= 0.72)
            matches = difflib.get_close_matches(t_clean, SCIENTIFIC_VOCABULARY, n=1, cutoff=0.72)
            if matches:
                clean_tokens.append(matches[0])
            else:
                clean_tokens.append(t_clean)
                
    clean_text = " ".join(clean_tokens)
    
    # Filter meaningful keywords (len > 2)
    stopwords = {'and', 'the', 'for', 'with', 'from', 'in', 'on', 'of', 'via', 'using', 'based', 'a', 'an'}
    keywords = [t for t in clean_tokens if len(t) > 2 and t not in stopwords]
    
    if not keywords:
        return clean_text, clean_text
        
    phrase = " ".join(clean_tokens)
    and_terms = " AND ".join([f"(ti:{t} OR abs:{t})" for t in keywords])
    
    # Target phrase in Title/Abstract OR all keywords in Title/Abstract
    structured_query = f'(ti:"{phrase}" OR abs:"{phrase}") OR ({and_terms})'
    return clean_text, structured_query


def search_arxiv(query: str, max_results: int = 500) -> List[Dict[str, Any]]:
    """
    Search arXiv for research papers matching the given query with high precision.

    Args:
        query (str): The search query (e.g., topic, keyword, or author).
        max_results (int, optional): The maximum number of results to fetch. Defaults to 500.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with paper metadata.
    """
    clean_text, structured_query = clean_and_normalize_query(query)
    logger.info(f"Searching arXiv for: '{clean_text}' (Query: {structured_query[:80]}...), max_results: {max_results}")
    
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=1.0, num_retries=3)
    
    # Attempt 1: High-precision structured query
    search = arxiv.Search(
        query=structured_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for result in client.results(search):
            paper = {
                "source": "arxiv",
                "external_id": result.get_short_id(),
                "title": result.title.replace('\n', ' ').strip(),
                "abstract": result.summary.replace('\n', ' ').strip(),
                "authors": [author.name for author in result.authors],
                "year": result.published.year if result.published else None,
                "pdf_url": result.pdf_url,
                "categories": result.categories
            }
            results.append(paper)
            
            if len(results) >= max_results:
                break
                
    except Exception as e:
        logger.warning(f"Structured search failed, falling back to clean text query: {e}")
        
    # Attempt 2: Fallback if structured query returned nothing
    if len(results) == 0:
        logger.info(f"Fallback to broad search on clean query: '{clean_text}'")
        search_fallback = arxiv.Search(
            query=clean_text,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        try:
            for result in client.results(search_fallback):
                paper = {
                    "source": "arxiv",
                    "external_id": result.get_short_id(),
                    "title": result.title.replace('\n', ' ').strip(),
                    "abstract": result.summary.replace('\n', ' ').strip(),
                    "authors": [author.name for author in result.authors],
                    "year": result.published.year if result.published else None,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories
                }
                results.append(paper)
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"Fallback arXiv search failed: {e}")
        
    logger.info(f"Successfully fetched {len(results)} relevant papers from arXiv.")
    return results

if __name__ == '__main__':
    # Standalone testing
    sample_query = "quantum computing"
    print(f"Testing arXiv scraper with query: '{sample_query}'")
    papers = search_arxiv(sample_query, max_results=5)
    for i, p in enumerate(papers, 1):
        print(f"\nResult {i}:")
        print(f"Title: {p['title']}")
        print(f"Authors: {', '.join(p['authors'])}")
        print(f"Year: {p['year']}")
        print(f"ID: {p['external_id']}")
        print(f"URL: {p['pdf_url']}")
