"""
Module for fetching papers via the Firecrawl research index.

Uses the Firecrawl v2 REST API directly (no SDK required):
  - GET /search/research/papers      — natural-language paper search
  - GET /search/research/papers/{id} — metadata + abstract for one paper

Auth: FIRECRAWL_API_KEY from the environment (or .env). The research
endpoints also answer without a key (rate-limited); the module works in
both cases and returns [] instead of raising when unreachable.
Docs: https://docs.firecrawl.dev/api-reference/v2-introduction
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.firecrawl.dev/v2"


def _load_env() -> None:
    for env_path in [".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]:
        if os.path.exists(env_path):
            with open(env_path) as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip())
            break


_load_env()


def _headers() -> Dict[str, str]:
    headers = {"User-Agent": "Academic Paper Recommender Course Project"}
    api_key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 2) -> Optional[Any]:
    """GET with light retry on 429/5xx. Returns parsed JSON or None."""
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=_headers(), params=params, timeout=30)
            if response.status_code == 200:
                time.sleep(0.5)
                return response.json()
            if response.status_code in (429, 500, 502, 503):
                logger.debug(f"Firecrawl {response.status_code}, retrying...")
                time.sleep(2.0 * (attempt + 1))
                continue
            logger.debug(f"Firecrawl {response.status_code}: {response.text[:150]}")
            return None
        except requests.exceptions.RequestException as exc:
            logger.debug(f"Firecrawl request failed: {exc}")
            time.sleep(2.0 * (attempt + 1))
    return None


def _first(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def _to_paper(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one research-index record to the pipeline paper schema."""
    authors = item.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",")]
    elif authors and isinstance(authors[0], dict):
        authors = [a.get("name") for a in authors if a.get("name")]
    links = item.get("links") or item.get("urls") or {}
    pdf_url = ""
    if isinstance(links, dict):
        pdf_url = links.get("pdf") or links.get("arxiv") or ""
    arxiv_id = _arxiv_id(item)
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    year = item.get("year") or item.get("publishedYear")
    if year is None:
        for key in ("createdDate", "updateDate", "publishedDate"):
            value = str(item.get(key) or "")
            if len(value) >= 4 and value[:4].isdigit():
                year = int(value[:4])
                break
    return {
        "source": "firecrawl",
        "external_id": str(_first(item.get("id"), item.get("paperId"), arxiv_id, item.get("doi")) or ""),
        "title": (item.get("title") or "").strip(),
        "abstract": (item.get("abstract") or item.get("summary") or "").strip(),
        "authors": [a for a in authors if a],
        "year": year,
        "pdf_url": pdf_url or item.get("url") or "",
        "categories": item.get("categories") or item.get("fieldsOfStudy") or [],
    }


def _inspect(paper_id: str, query: str = "") -> Dict[str, Any]:
    """Fetch full metadata/abstract for one paper; {} on failure."""
    params = {"query": query} if query else None
    data = _get(f"{BASE_URL}/search/research/papers/{paper_id}", params=params)
    if isinstance(data, dict):
        for key in ("paper", "data", "result"):
            if isinstance(data.get(key), dict):
                return data[key]
        if data.get("title"):
            return data
    return {}


def _arxiv_id(item: Dict[str, Any]) -> str:
    ids = item.get("ids") or {}
    if isinstance(ids, dict):
        arxiv_ids = ids.get("arxiv") or []
        if arxiv_ids:
            return str(arxiv_ids[0])
    primary = str(item.get("primaryId") or "")
    if primary.lower().startswith("arxiv:"):
        return primary.split(":", 1)[1]
    return ""


def search_firecrawl(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    """Search Firecrawl's paper index and return normalized paper dicts.

    Each search hit is inspected individually so results carry abstracts,
    authors, year and PDF links — not just titles.
    """
    logger.info(f"Searching Firecrawl research index for: '{query}', max_results: {max_results}")
    # NOTE: the endpoint takes no paging params; slice client-side.
    data = _get(f"{BASE_URL}/search/research/papers", params={"query": query})
    if not data:
        return []
    items = data.get("results") or data.get("data") or data.get("papers") or []
    if isinstance(items, dict):
        items = list(items.values())

    papers: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        paper_id = _first(item.get("id"), item.get("paperId"))
        merged = dict(item)
        # Search hits carry no authors/year, so always inspect for full metadata.
        if paper_id:
            detail = _inspect(str(paper_id))
            if detail:
                merged.update(detail)
        paper = _to_paper(merged)
        if paper["title"]:
            papers.append(paper)
        if len(papers) >= max_results:
            break

    logger.info(f"Fetched {len(papers)} papers from Firecrawl.")
    return papers


if __name__ == "__main__":
    for paper in search_firecrawl("attention mechanism transformers", max_results=3):
        print(f"- {paper['title'][:80]} ({paper['year']}) [{paper['external_id']}]")
