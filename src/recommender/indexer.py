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
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

logger = logging.getLogger(__name__)


class PaperIndex:
    """Manages the FAISS vector index for paper embeddings."""

    def __init__(
        self,
        model_path: str = "models/scibert-finetuned-papers",
        index_path: str = "data/faiss_index.bin",
        mapping_path: str = "data/paper_id_map.json",
        pooling: str = "mean",
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
        self.pooling = pooling

        import warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # A missing local checkpoint must NOT be passed to SentenceTransformer:
        # it would be treated as a Hugging Face repo ID and fail with a 401.
        if os.path.isdir(model_path):
            try:
                self.model = SentenceTransformer(os.path.abspath(model_path))
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}, using the untrained {pooling} profile. Error: {e}")
                from src.model.train import create_model
                self.model = create_model('allenai/scibert_scivocab_uncased', use_qlora=False, use_lora=False, pooling=pooling)
        else:
            logger.info(f"No local checkpoint at {model_path}; using base allenai/scibert_scivocab_uncased with {pooling} pooling.")
            from src.model.train import create_model
            self.model = create_model('allenai/scibert_scivocab_uncased', use_qlora=False, use_lora=False, pooling=pooling)
            
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index and mapping
        self.index = faiss.IndexFlatIP(self.dimension)
        self.mapping: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []
        
        self.load()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize scientific text consistently for BM25."""
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())

    def _rebuild_bm25(self) -> None:
        corpus = [f"{item.get('title') or ''} {item.get('abstract') or ''}" for item in self.mapping]
        self._bm25_tokens = [self._tokenize(text) for text in corpus]
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

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
            # External providers may return null for optional metadata. Normalize
            # it before embedding and persisting so every indexed field is text.
            title = str(paper.get("title") or "").strip()
            title_lower = title.lower()
            
            if (ext_id and ext_id in existing_ids) or (title_lower and title_lower in existing_titles):
                continue
                
            if ext_id:
                existing_ids.add(ext_id)
            if title_lower:
                existing_titles.add(title_lower)
                
            abstract = str(paper.get("abstract") or "").strip()
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
        self._rebuild_bm25()
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
        
        candidate_k = min(max(top_k * 10, 100), self.index.ntotal)
        dense_scores, dense_indices = self.index.search(embedding, candidate_k)

        if self._bm25 is None:
            self._rebuild_bm25()
        bm25_all = self._bm25.get_scores(self._tokenize(query_text)) if self._bm25 else np.zeros(len(self.mapping))
        lexical_indices = np.argsort(-bm25_all)[:candidate_k]

        # Fuse the dense and lexical candidate sets before scoring. This
        # recovers papers with exact scientific terminology that a dense-only
        # search can miss while retaining semantically related results.
        dense_by_index = {int(index): float(score) for score, index in zip(dense_scores[0], dense_indices[0]) if index >= 0}
        candidate_indices = list(dict.fromkeys([*dense_by_index, *map(int, lexical_indices)]))
        lexical_values = np.asarray([bm25_all[index] for index in candidate_indices], dtype=np.float32)
        lexical_min = float(lexical_values.min()) if len(lexical_values) else 0.0
        lexical_range = float(np.ptp(lexical_values)) if len(lexical_values) else 0.0
        
        candidates = []
        seen_titles = set()
        
        for idx in candidate_indices:
            if idx != -1 and idx < len(self.mapping):
                result = self.mapping[idx].copy()
                # Older mapping files can already contain null values, so keep
                # search compatible with them as well as newly indexed papers.
                title = str(result.get('title') or '').strip()
                title_lower = title.lower()
                
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)
                
                dense_score = dense_by_index.get(idx)
                if dense_score is None:
                    dense_score = float(np.dot(embedding[0], self.index.reconstruct(idx)))
                lexical_score = (float(bm25_all[idx]) - lexical_min) / lexical_range if lexical_range > 0 else 0.0
                normalized_dense = max(0.0, min(1.0, (dense_score + 1.0) / 2.0))
                combined_score = (0.35 * normalized_dense) + (0.65 * lexical_score)
                
                result["score"] = combined_score
                result["raw_dense_score"] = dense_score
                result["bm25_score"] = float(bm25_all[idx])
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
                self._rebuild_bm25()
                logger.info(f"Loaded index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading index/mapping: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.mapping = []
        else:
            logger.info("Index or mapping not found. Initialized empty index.")
