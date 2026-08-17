"""
Module for scraping papers from Semantic Scholar.

This module provides functions to search Semantic Scholar using its Graph API,
as well as functions to retrieve citations and references for specific papers.
"""

import os
import time
import logging
import requests
from typing import List, Dict, Any

# Load .env file if present
def _load_env():
    """Load variables from .env file in project root."""
    for env_path in ['.env', os.path.join(os.path.dirname(__file__), '..', '..', '.env')]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        os.environ.setdefault(key.strip(), value.strip())
            break

_load_env()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
def _get_headers() -> Dict[str, str]:
    """Return headers for the Semantic Scholar API request."""
    _load_env()
    api_key = (os.environ.get("S2_API_KEY") or "").strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Academic Paper Recommender Course Project)"
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers

def _make_request(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Make an HTTP GET request to the given URL with exponential backoff for 429/temporary errors.
    
    Args:
        url (str): The endpoint URL.
        params (Dict[str, Any], optional): The query parameters.
        
    Returns:
        Dict[str, Any]: The JSON response or empty dict on failure.
    """
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=_get_headers(), params=params, timeout=15)
            
            if response.status_code == 200:
                time.sleep(1.0)  # rate limiting polite delay
                return response.json()
            elif response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                logger.debug(f"Semantic Scholar rate limited (429). Retrying in {delay}s...")
                time.sleep(delay)
            elif response.status_code == 403:
                logger.info("Semantic Scholar API key is inactive or restricted (403). Using arXiv open access corpus.")
                return {}
            else:
                logger.debug(f"Status {response.status_code} from Semantic Scholar: {response.text[:100]}")
                time.sleep(base_delay)
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logger.warning(f"Could not connect to Semantic Scholar: {e}. Skipping.")
                return {}
            time.sleep(base_delay * (2 ** attempt))
                
    return {}

def search_semantic_scholar(query: str, max_results: int = 500) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers matching the given query.

    Args:
        query (str): The search query.
        max_results (int, optional): The maximum number of results to fetch. Defaults to 500.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the scraped papers.
    """
    logger.info(f"Searching Semantic Scholar for query: '{query}', max_results: {max_results}")
    
    url = f"{BASE_URL}/paper/search"
    fields = "paperId,title,abstract,year,citationCount,url,authors"
    
    results = []
    offset = 0
    limit = 100
    
    while len(results) < max_results:
        current_limit = min(limit, max_results - len(results))
        params = {
            "query": query,
            "fields": fields,
            "offset": offset,
            "limit": current_limit
        }
        
        try:
            data = _make_request(url, params)
            if not data or "data" not in data or not data["data"]:
                break
                
            for item in data["data"]:
                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
                
                paper = {
                    "source": "semantic_scholar",
                    "external_id": item.get("paperId"),
                    "title": item.get("title"),
                    "abstract": item.get("abstract"),
                    "authors": authors,
                    "year": item.get("year"),
                    "pdf_url": item.get("url"),
                    "categories": [],
                    "citation_ids": [],
                    "reference_ids": []
                }
                results.append(paper)
                
                if len(results) >= max_results:
                    break
                    
            logger.info(f"Fetched {len(results)} papers from Semantic Scholar...")
            offset += limit
            
            if "next" not in data:
                break
                
        except Exception as e:
            logger.error(f"Error during Semantic Scholar search: {e}")
            break
            
    logger.info(f"Successfully fetched {len(results)} papers from Semantic Scholar.")
    return results

def get_paper_citations_and_references(paper_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve citations and references for a specific paper.

    Args:
        paper_id (str): The Semantic Scholar paper ID.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with 'citations' and 'references' lists.
    """
    logger.debug(f"Fetching citations and references for paper: {paper_id}")
    result = {"citations": [], "references": []}
    
    fields = "paperId,title,abstract,year,url"
    
    # Fetch citations
    cit_url = f"{BASE_URL}/paper/{paper_id}/citations"
    try:
        data = _make_request(cit_url, {"fields": fields, "limit": 100})
        if data and "data" in data:
            result["citations"] = [item.get("citingPaper", {}) for item in data["data"] if item.get("citingPaper")]
    except Exception as e:
        logger.error(f"Error fetching citations for {paper_id}: {e}")
        
    # Fetch references
    ref_url = f"{BASE_URL}/paper/{paper_id}/references"
    try:
        data = _make_request(ref_url, {"fields": fields, "limit": 100})
        if data and "data" in data:
            result["references"] = [item.get("citedPaper", {}) for item in data["data"] if item.get("citedPaper")]
    except Exception as e:
        logger.error(f"Error fetching references for {paper_id}: {e}")
        
    return result

def enrich_with_citations(papers: List[Dict[str, Any]], max_papers: int = 100) -> List[Dict[str, Any]]:
    """
    Enrich a list of Semantic Scholar papers with citation and reference IDs.

    Args:
        papers (List[Dict[str, Any]]): The list of papers to enrich.
        max_papers (int, optional): The maximum number of papers to enrich. Defaults to 100.

    Returns:
        List[Dict[str, Any]]: The enriched list of papers.
    """
    logger.info(f"Enriching up to {max_papers} papers with citations and references...")
    
    enriched = []
    count = 0
    
    for paper in papers:
        if count >= max_papers:
            enriched.append(paper)
            continue
            
        if paper.get("source") != "semantic_scholar" or not paper.get("external_id"):
            enriched.append(paper)
            continue
            
        try:
            cit_ref_data = get_paper_citations_and_references(paper["external_id"])
            
            cit_ids = [c.get("paperId") for c in cit_ref_data.get("citations", []) if c.get("paperId")]
            ref_ids = [r.get("paperId") for r in cit_ref_data.get("references", []) if r.get("paperId")]
            
            paper["citation_ids"] = cit_ids
            paper["reference_ids"] = ref_ids
            
            count += 1
            if count % 10 == 0:
                logger.info(f"Enriched {count} papers...")
                
        except Exception as e:
            logger.error(f"Failed to enrich paper {paper.get('external_id')}: {e}")
            
        enriched.append(paper)
        
    logger.info(f"Successfully enriched {count} papers.")
    return enriched

if __name__ == '__main__':
    # Standalone testing
    sample_query = "transformers NLP"
    print(f"Testing Semantic Scholar scraper with query: '{sample_query}'")
    papers = search_semantic_scholar(sample_query, max_results=2)
    
    for i, p in enumerate(papers, 1):
        print(f"\nResult {i}:")
        print(f"Title: {p['title']}")
        print(f"Authors: {', '.join(p['authors'])}")
        print(f"Year: {p['year']}")
        print(f"ID: {p['external_id']}")
        
    if papers:
        print("\nTesting enrichment on the first result...")
        enriched = enrich_with_citations([papers[0]], max_papers=1)
        print(f"Found {len(enriched[0].get('citation_ids', []))} citations and {len(enriched[0].get('reference_ids', []))} references.")
