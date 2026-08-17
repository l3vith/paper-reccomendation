"""
Scientific Query Expansion Module using NLTK and BM25 Pseudo-Relevance Feedback (PRF).

Combines:
1. NLTK (WordNet Synsets, Lemmatization, Stopword Filtering) for linguistic expansion.
2. BM25Okapi (Pseudo-Relevance Feedback / RM3) for corpus-grounded IR expansion.
3. Technical Acronym Resolution for scientific and ML domains.
"""

import os
import re
import sqlite3
import logging
from typing import List, Dict, Set, Optional, Tuple

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Ensure NLTK data is ready
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

# Technical Acronym Knowledge Base
ACRONYM_MAP = {
    # NLP & LLMs
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "rag": "retrieval augmented generation",
    "rlhf": "reinforcement learning from human feedback",
    "cot": "chain of thought reasoning",
    "ner": "named entity recognition",
    "pos": "part of speech tagging",
    "nli": "natural language inference",
    "qa": "question answering",
    "mt": "machine translation",
    "bert": "bidirectional encoder representations transformers",
    "gpt": "generative pretrained transformer",
    "peft": "parameter efficient fine tuning",
    "lora": "low rank adaptation",
    "qlora": "quantized low rank adaptation",
    "mamba": "state space models sequence modeling",
    "ssm": "state space models",
    
    # Machine Learning & AI
    "ml": "machine learning",
    "dl": "deep learning",
    "gnn": "graph neural networks",
    "gnns": "graph neural networks",
    "cnn": "convolutional neural networks",
    "cnns": "convolutional neural networks",
    "rnn": "recurrent neural networks",
    "rnns": "recurrent neural networks",
    "vae": "variational autoencoders",
    "gan": "generative adversarial networks",
    "gans": "generative adversarial networks",
    "vit": "vision transformers",
    "cv": "computer vision",
    "kd": "knowledge distillation",
    
    # Biology & Science
    "ppi": "protein protein interaction",
    "go": "gene ontology",
    "msa": "multiple sequence alignment",
    "pdb": "protein data bank structural biology",
    "crispr": "clustered regularly interspaced short palindromic repeats",
    "scrna": "single cell rna sequencing",
    "eqtl": "expression quantitative trait loci",
    "gwas": "genome wide association study",
    
    # Quantum
    "qml": "quantum machine learning",
    "qaoa": "quantum approximate optimization algorithm",
    "vqe": "variational quantum eigensolver",
    "nisq": "noisy intermediate scale quantum",
}

# Base stopwords for scientific IR
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english')).union({
        "research", "paper", "papers", "study", "studies", "approach", "approaches",
        "method", "methods", "model", "models", "system", "systems", "result", "results",
        "using", "based", "novel", "towards", "via", "presents", "proposed", "show",
        "demonstrate", "also", "two", "three", "first", "data", "performance", "task"
    })
except Exception:
    ENGLISH_STOPWORDS = {
        "and", "the", "for", "with", "from", "in", "on", "of", "to", "a", "an", "is",
        "research", "paper", "papers", "study", "method", "system", "using", "based"
    }


class ScientificQueryExpander:
    """
    Standard Library-Driven Scientific Query Expander.
    Leverages NLTK WordNet and Rank-BM25 Pseudo-Relevance Feedback (PRF).
    """

    def __init__(self, db_path: str = "data/papers.db"):
        self.db_path = db_path
        self.lemmatizer = WordNetLemmatizer()
        self._bm25_model: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        self._corpus_loaded = False

    def _init_bm25_corpus(self) -> None:
        """Load SQLite abstracts and initialize the BM25 Okapi model for Pseudo-Relevance Feedback."""
        if self._corpus_loaded or not os.path.exists(self.db_path):
            return

        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT title, abstract FROM papers WHERE title IS NOT NULL OR abstract IS NOT NULL")
            rows = c.fetchall()
            conn.close()

            tokenized = []
            for title, abstract in rows:
                text = f"{title or ''} {abstract or ''}"
                raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                doc_tokens = [
                    self.lemmatizer.lemmatize(w)
                    for w in raw_words
                    if w not in ENGLISH_STOPWORDS
                ]
                if doc_tokens:
                    tokenized.append(doc_tokens)

            if tokenized:
                self._tokenized_corpus = tokenized
                self._bm25_model = BM25Okapi(tokenized)
                self._corpus_loaded = True
                logger.debug(f"BM25 PRF corpus initialized with {len(tokenized)} documents.")
        except Exception as e:
            logger.debug(f"BM25 initialization skipped: {e}")

    def _resolve_acronyms(self, query: str) -> List[str]:
        """Identify scientific acronyms and expand them using NLTK token boundaries."""
        tokens = re.findall(r'\b[A-Za-z]+\b', query.lower())
        expansions = []
        for t in tokens:
            if t in ACRONYM_MAP:
                expansions.append(ACRONYM_MAP[t])
        return expansions

    def _expand_with_wordnet(self, query: str, max_synonyms: int = 2) -> List[str]:
        """Extract domain-relevant semantic lemmas from NLTK WordNet."""
        raw_words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        synonyms = []
        
        for w in raw_words:
            if w in ENGLISH_STOPWORDS:
                continue
            lem_w = self.lemmatizer.lemmatize(w)
            synsets = wn.synsets(lem_w)
            for syn in synsets[:1]:  # Primary synset only
                for lemma in syn.lemmas()[:2]:
                    name = lemma.name().lower()
                    # Only accept clean single-word lemmas that are not stopwords or trivial
                    if ('_' not in name and 
                        name != w and 
                        name != lem_w and 
                        name not in ENGLISH_STOPWORDS and 
                        len(name) > 3):
                        synonyms.append(name)
                        if len(synonyms) >= max_synonyms:
                            return list(dict.fromkeys(synonyms))

        return list(dict.fromkeys(synonyms))

    def _expand_with_bm25_prf(self, query: str, top_k_docs: int = 5, top_terms: int = 3) -> List[str]:
        """
        Extract informative expansion keywords via BM25 Pseudo-Relevance Feedback (PRF / RM3).
        Calculates term relevance weights across the top pseudo-relevant corpus documents.
        """
        self._init_bm25_corpus()
        if not self._bm25_model or not self._tokenized_corpus:
            return []

        raw_words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        q_tokens = [self.lemmatizer.lemmatize(w) for w in raw_words if w not in ENGLISH_STOPWORDS]

        if not q_tokens:
            return []

        try:
            scores = self._bm25_model.get_scores(q_tokens)
            top_doc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_docs]

            term_scores: Dict[str, float] = {}
            for idx in top_doc_indices:
                score = scores[idx]
                if score <= 0:
                    continue
                doc_tokens = self._tokenized_corpus[idx]
                doc_len = len(doc_tokens) + 1
                for token in doc_tokens:
                    if token not in q_tokens and token not in ENGLISH_STOPWORDS and len(token) > 3:
                        term_scores[token] = term_scores.get(token, 0.0) + (score / doc_len)

            # Sort terms by PRF relevance
            ranked_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            return [term for term, _ in ranked_terms[:top_terms]]
        except Exception as e:
            logger.debug(f"BM25 PRF expansion skipped: {e}")
            return []

    def expand(self, query: str, max_additional_terms: int = 4) -> Dict[str, any]:
        """
        Execute multi-stage query expansion with NLTK and BM25 PRF.

        Args:
            query (str): The search topic or raw query string.
            max_additional_terms (int): Max number of distinct expansion terms to include.

        Returns:
            dict containing:
              - 'original_query': Original input query.
              - 'acronym_expansions': Acronym expansions.
              - 'wordnet_synonyms': Synonyms from NLTK WordNet.
              - 'bm25_prf_terms': Pseudo-Relevance Feedback terms from BM25.
              - 'all_expansion_terms': Combined deduplicated expansion terms.
              - 'dense_query': Dense query string for SentenceTransformer vector search.
              - 'scraper_query': Precision query string for live API scraping.
        """
        clean_query = query.strip()

        # 1. Linguistic normalization with NLTK
        tokens = re.findall(r'\b[a-zA-Z]+\b', clean_query.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        normalized_query = " ".join(lemmatized_tokens)

        # 2. Acronym resolution
        acronyms = self._resolve_acronyms(clean_query)

        # 3. BM25 Pseudo-Relevance Feedback (PRF)
        bm25_terms = self._expand_with_bm25_prf(clean_query, top_k_docs=5, top_terms=3)

        # 4. NLTK WordNet Synsets
        wordnet_synonyms = self._expand_with_wordnet(clean_query, max_synonyms=2)

        # Combine unique expansions in order of priority: Acronyms -> BM25 PRF -> WordNet
        combined: List[str] = []
        seen: Set[str] = set()

        for source_list in [acronyms, bm25_terms, wordnet_synonyms]:
            for term in source_list:
                term_clean = term.lower().strip()
                if (term_clean not in seen and 
                    term_clean not in clean_query.lower() and 
                    term_clean not in ENGLISH_STOPWORDS and 
                    len(term_clean) > 2):
                    seen.add(term_clean)
                    combined.append(term_clean)

        selected_expansions = combined[:max_additional_terms]

        # Construct Dense Query for FAISS vector search
        if selected_expansions:
            dense_query = f"{clean_query} ({', '.join(selected_expansions)})"
        else:
            dense_query = clean_query

        # Construct Scraper Query for arXiv API
        if selected_expansions:
            scraper_query = f'{clean_query} OR ("{" ".join(selected_expansions[:2])}")'
        else:
            scraper_query = clean_query

        return {
            "original_query": clean_query,
            "normalized_query": normalized_query,
            "acronym_expansions": acronyms,
            "wordnet_synonyms": wordnet_synonyms,
            "bm25_prf_terms": bm25_terms,
            "all_expansion_terms": selected_expansions,
            "dense_query": dense_query,
            "scraper_query": scraper_query,
        }
