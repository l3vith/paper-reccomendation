"""
Builds and manages a FAISS vector index for paper embeddings.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PaperIndex:
    """Manages the FAISS vector index for paper embeddings."""

    def __init__(
        self,
        model_path: str = "models/scibert-finetuned-papers",
        index_path: str = "data/faiss_index.bin",
        mapping_path: str = "data/paper_id_map.json"
    ) -> None:
        """
        Initialize the PaperIndex.

        Args:
            model_path: Path to the sentence-transformers model.
            index_path: Path to save/load the FAISS index.
            mapping_path: Path to save/load the paper ID mapping.
        """
        self.model_path = model_path
        self.index_path = index_path
        self.mapping_path = mapping_path

        import warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Resolve local absolute path if directory exists
        load_target = os.path.abspath(model_path) if os.path.isdir(model_path) else model_path

        try:
            self.model = SentenceTransformer(load_target)
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}, using default. Error: {e}")
            self.model = SentenceTransformer('allenai/scibert_scivocab_uncased')
            
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index and mapping
        self.index = faiss.IndexFlatIP(self.dimension)
        self.mapping: List[Dict[str, Any]] = []
        
        self.load()

    def build_index(self, db_path: str = "data/papers.db") -> int:
        """
        Build index from a SQLite database.

        Args:
            db_path: Path to the SQLite database.

        Returns:
            Number of papers indexed.
        """
        logger.info(f"Building index from {db_path}...")
        
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return 0
            
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Assuming a table named 'papers'
            cursor.execute("SELECT * FROM papers")
            rows = cursor.fetchall()
            
            papers_to_add = []
            for row in rows:
                paper = dict(row)
                papers_to_add.append(paper)
                
            conn.close()
            
            if not papers_to_add:
                logger.info("No papers found in database.")
                return 0
                
            # Clear existing index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.mapping = []
            
            return self.add_papers(papers_to_add)
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return 0

    def add_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        Add new papers to the index, ignoring duplicates.

        Args:
            papers: List of paper dictionaries.

        Returns:
            Number of new papers added.
        """
        if not papers:
            return 0
            
        existing_ids = {m.get("external_id") for m in self.mapping if m.get("external_id")}
        existing_titles = {m.get("title", "").strip().lower() for m in self.mapping if m.get("title")}
        
        texts = []
        new_mappings = []
        
        for paper in papers:
            ext_id = paper.get("external_id")
            title = (paper.get("title") or "").strip()
            title_lower = title.lower()
            
            if (ext_id and ext_id in existing_ids) or (title_lower and title_lower in existing_titles):
                continue
                
            if ext_id:
                existing_ids.add(ext_id)
            if title_lower:
                existing_titles.add(title_lower)
                
            abstract = paper.get("abstract", "")
            text = f"{title} [SEP] {abstract}"
            texts.append(text)
            
            new_mappings.append({
                "id": paper.get("id"),
                "title": title,
                "abstract": abstract,
                "pdf_url": paper.get("pdf_url"),
                "year": paper.get("year"),
                "source": paper.get("source"),
                "external_id": ext_id
            })
            
        if not texts:
            logger.info("All provided papers are already in index. No new embeddings added.")
            return 0

        logger.info(f"Encoding and indexing {len(texts)} new unique papers...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.mapping.extend(new_mappings)
        self.save()
        return len(texts)

    def search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index for nearest neighbors with hybrid semantic & keyword reranking and deduplication.

        Args:
            query_text: Text to query.
            top_k: Number of results to return.

        Returns:
            List of result dictionaries.
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty. Cannot search.")
            return []
            
        embedding = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        
        candidate_k = min(max(top_k * 4, 30), self.index.ntotal)
        scores, indices = self.index.search(embedding, candidate_k)
        
        stopwords = {'and', 'the', 'for', 'with', 'from', 'in', 'on', 'of', 'a', 'an', 'to', 'is', 'are', 'as', 'by', 'using', 'via'}
        query_words = [w.lower() for w in query_text.split() if len(w) > 2 and w.lower() not in stopwords]
        
        candidates = []
        seen_titles = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.mapping):
                result = self.mapping[idx].copy()
                title = result.get('title', '').strip()
                title_lower = title.lower()
                
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)
                
                dense_score = float(score)
                abstract_lower = result.get('abstract', '').lower()
                
                # Check keyword matches
                title_matches = sum(1 for w in query_words if w in title_lower)
                abstract_matches = sum(1 for w in query_words if w in abstract_lower)
                
                # Keyword relevance weighting
                keyword_boost = (title_matches * 0.12) + (abstract_matches * 0.04)
                combined_score = min(dense_score + keyword_boost, 0.99)
                
                result["score"] = combined_score
                result["raw_dense_score"] = dense_score
                candidates.append(result)
                
        # Sort by combined hybrid score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        final_results = []
        for i, res in enumerate(candidates[:top_k]):
            res["rank"] = i + 1
            final_results.append(res)
            
        return final_results

    def save(self) -> None:
        """Save the index and mapping to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index ({self.index.ntotal} vectors) to {self.index_path} and mapping to {self.mapping_path}")

    def load(self) -> None:
        """Load the index and mapping from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                logger.info(f"Loaded index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading index/mapping: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.mapping = []
        else:
            logger.info("Index or mapping not found. Initialized empty index.")
