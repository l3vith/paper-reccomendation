"""
Module for building training triplets from the citation graph stored in SQLite.
"""

import sqlite3
import json
import random
import logging
import argparse
from typing import List, Tuple, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_papers(db_path: str = 'data/papers.db') -> List[Dict[str, Any]]:
    """
    Load all papers from the database.
    
    Args:
        db_path (str): Path to the SQLite database.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict represents a paper.
    """
    logging.info(f"Loading papers from {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM papers")
    rows = cursor.fetchall()
    
    papers = []
    for row in rows:
        paper = dict(row)
        # Parse JSON fields
        for field in ['authors', 'categories', 'citation_ids', 'reference_ids']:
            if field in paper and paper[field]:
                try:
                    paper[field] = json.loads(paper[field])
                except json.JSONDecodeError:
                    paper[field] = []
            else:
                paper[field] = []
        papers.append(paper)
        
    conn.close()
    logging.info(f"Loaded {len(papers)} papers")
    return papers

def build_paper_text(paper: Dict[str, Any]) -> str:
    """
    Concatenate paper title and abstract.
    
    Args:
        paper (Dict[str, Any]): Paper dictionary.
        
    Returns:
        str: Concatenated text or empty string if both are missing.
    """
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    
    if not title and not abstract:
        return ""
    
    parts = []
    if title:
        parts.append(title.strip())
    if abstract:
        parts.append(abstract.strip())
        
    return ' [SEP] '.join(parts)

def generate_triplets(papers: List[Dict[str, Any]], max_positives_per_anchor: int = 5, hard_negatives: bool = True) -> List[Tuple[str, str, str]]:
    """
    Generate triplets (anchor, positive, negative) for training.
    
    Args:
        papers (List[Dict[str, Any]]): List of papers.
        max_positives_per_anchor (int): Max positive examples per anchor.
        hard_negatives (bool): Whether to use hard negatives (same category).
        
    Returns:
        List[Tuple[str, str, str]]: List of generated triplets.
    """
    logging.info("Building lookup dictionary")
    # Using 'external_id' as required, assuming it is the ID used in references/citations
    lookup = {paper.get('external_id', paper.get('id')): paper for paper in papers if paper.get('external_id') or paper.get('id')}
    
    # Pre-compute categories to speed up hard negative mining
    category_to_papers = {}
    if hard_negatives:
        for p_id, p in lookup.items():
            for cat in p.get('categories', []):
                if cat not in category_to_papers:
                    category_to_papers[cat] = []
                category_to_papers[cat].append(p_id)
                
    all_paper_ids = list(lookup.keys())
    
    triplets = []
    anchors_with_citations = 0
    
    logging.info("Generating triplets")
    for paper in papers:
        citation_ids = set(paper.get('citation_ids', []) + paper.get('reference_ids', []))
        if not citation_ids:
            continue
            
        anchor_text = build_paper_text(paper)
        if not anchor_text:
            continue
            
        valid_positives = [lookup[c_id] for c_id in citation_ids if c_id in lookup]
        if not valid_positives:
            continue
            
        anchors_with_citations += 1
        
        # Limit positives
        random.shuffle(valid_positives)
        valid_positives = valid_positives[:max_positives_per_anchor]
        
        for pos_paper in valid_positives:
            pos_text = build_paper_text(pos_paper)
            if not pos_text:
                continue
                
            # Find a negative example
            neg_paper_id = None
            if hard_negatives and paper.get('categories'):
                # Try to find from same category
                cat = random.choice(paper.get('categories'))
                cat_papers = category_to_papers.get(cat, [])
                candidates = [pid for pid in cat_papers if pid not in citation_ids and pid != paper.get('external_id', paper.get('id'))]
                if candidates:
                    neg_paper_id = random.choice(candidates)
            
            if not neg_paper_id:
                # Fallback to random negative
                while True:
                    neg_paper_id = random.choice(all_paper_ids)
                    if neg_paper_id not in citation_ids and neg_paper_id != paper.get('external_id', paper.get('id')):
                        break
                        
            neg_paper = lookup[neg_paper_id]
            neg_text = build_paper_text(neg_paper)
            
            if neg_text:
                triplets.append((anchor_text, pos_text, neg_text))
                
    logging.info(f"Total anchors with citations: {anchors_with_citations}")
    logging.info(f"Total triplets generated from citations: {len(triplets)}")
    
    # Fallback: If no/few citation triplets generated (e.g. arXiv-only data), mine by category similarity
    if len(triplets) < 50 and len(papers) >= 5:
        logging.info("Citation graph sparse. Mining triplets using category co-occurrence and topical clustering...")
        for paper in papers:
            anchor_text = build_paper_text(paper)
            if not anchor_text:
                continue
            
            paper_cats = set(paper.get('categories', []))
            p_id = paper.get('external_id', paper.get('id'))
            
            # Find positive candidates (papers sharing at least one category)
            pos_candidates = []
            if paper_cats:
                for cat in paper_cats:
                    for cid in category_to_papers.get(cat, []):
                        if cid != p_id and cid not in pos_candidates:
                            pos_candidates.append(cid)
            
            if not pos_candidates:
                # If no categories, take random from corpus as mild positive
                pos_candidates = [pid for pid in all_paper_ids if pid != p_id]
                
            random.shuffle(pos_candidates)
            selected_pos = pos_candidates[:max_positives_per_anchor]
            
            for pos_id in selected_pos:
                pos_paper = lookup[pos_id]
                pos_text = build_paper_text(pos_paper)
                if not pos_text:
                    continue
                
                # Negative candidates (papers with disjoint categories or random other papers)
                neg_candidates = [
                    pid for pid in all_paper_ids 
                    if pid != p_id and pid != pos_id and not (set(lookup[pid].get('categories', [])) & paper_cats)
                ]
                if not neg_candidates:
                    neg_candidates = [pid for pid in all_paper_ids if pid != p_id and pid != pos_id]
                
                if neg_candidates:
                    neg_paper = lookup[random.choice(neg_candidates)]
                    neg_text = build_paper_text(neg_paper)
                    if neg_text:
                        triplets.append((anchor_text, pos_text, neg_text))
                        
        logging.info(f"Total triplets generated after category mining: {len(triplets)}")

    return triplets

def save_triplets(triplets: List[Tuple[str, str, str]], output_path: str = 'data/triplets.csv') -> None:
    """
    Save triplets to a CSV file.
    
    Args:
        triplets (List[Tuple[str, str, str]]): List of triplets to save.
        output_path (str): Path to output CSV.
    """
    logging.info(f"Saving triplets to {output_path}")
    df = pd.DataFrame(triplets, columns=['anchor_text', 'positive_text', 'negative_text'])
    df.to_csv(output_path, index=False)
    logging.info("Saved successfully")

def load_triplets(path: str = 'data/triplets.csv') -> List[Tuple[str, str, str]]:
    """
    Load triplets from a CSV file.
    
    Args:
        path (str): Path to CSV file.
        
    Returns:
        List[Tuple[str, str, str]]: List of loaded triplets.
    """
    logging.info(f"Loading triplets from {path}")
    df = pd.read_csv(path)
    triplets = [tuple(x) for x in df.to_numpy()]
    logging.info(f"Loaded {len(triplets)} triplets")
    return triplets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build training triplets from citation graph")
    parser.add_argument('--db-path', type=str, default='data/papers.db', help='Path to input SQLite database')
    parser.add_argument('--output', type=str, default='data/triplets.csv', help='Path to output CSV file')
    parser.add_argument('--max-positives', type=int, default=5, help='Max positive examples per anchor')
    parser.add_argument('--hard-negatives', action='store_true', help='Use hard negatives')
    
    args = parser.parse_args()
    
    papers = load_papers(args.db_path)
    triplets = generate_triplets(papers, args.max_positives, args.hard_negatives)
    save_triplets(triplets, args.output)
    
    print("\nSummary:")
    print(f"Total papers: {len(papers)}")
    print(f"Triplets generated: {len(triplets)}")
