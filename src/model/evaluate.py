"""
Module for evaluating the fine-tuned model against vanilla SciBERT.
"""

import logging
import argparse
import json
import random
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, util
from .build_triplets import load_papers, build_paper_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_triplet_accuracy(model: SentenceTransformer, test_triplets: List[Tuple[str, str, str]], batch_size: int = 32) -> float:
    """
    Compute accuracy on triplet task: cosine_sim(anchor, positive) > cosine_sim(anchor, negative).
    
    Args:
        model (SentenceTransformer): The model to evaluate.
        test_triplets (List[Tuple[str, str, str]]): List of test triplets (anchor, pos, neg).
        batch_size (int): Batch size for encoding.
        
    Returns:
        float: Fraction of correct triplets.
    """
    logging.info("Computing triplet accuracy...")
    anchors = [t[0] for t in test_triplets]
    positives = [t[1] for t in test_triplets]
    negatives = [t[2] for t in test_triplets]
    
    # Encode all texts in batches
    anchor_embs = model.encode(anchors, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    pos_embs = model.encode(positives, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    neg_embs = model.encode(negatives, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute cosine similarities
    sim_pos = util.cos_sim(anchor_embs, pos_embs).diagonal()
    sim_neg = util.cos_sim(anchor_embs, neg_embs).diagonal()
    
    # Calculate accuracy
    correct = (sim_pos > sim_neg).sum().item()
    accuracy = correct / len(test_triplets) if test_triplets else 0.0
    
    logging.info(f"Triplet Accuracy: {accuracy:.4f}")
    return accuracy

def compute_mrr_and_precision(model: SentenceTransformer, papers: List[Dict[str, Any]], k_values: List[int] = [5, 10], sample_size: int = 100) -> Dict[str, float]:
    """
    Compute MRR and Precision@K for a sample of papers.
    
    Args:
        model (SentenceTransformer): The model to evaluate.
        papers (List[Dict[str, Any]]): List of all papers.
        k_values (List[int]): K values for Precision.
        sample_size (int): Number of papers to sample for evaluation.
        
    Returns:
        Dict[str, float]: Dictionary containing MRR and Precision metrics.
    """
    logging.info("Computing MRR and Precision...")
    
    # Filter papers with valid text and citations
    valid_papers = []
    paper_texts = []
    paper_ids = []
    
    for p in papers:
        text = build_paper_text(p)
        if text:
            valid_papers.append(p)
            paper_texts.append(text)
            paper_ids.append(p.get('external_id', p.get('id')))
            
    # Sample queries (papers with citations)
    queries = []
    for p in valid_papers:
        cites = set(p.get('citation_ids', []) + p.get('reference_ids', []))
        # Keep only citations that are in our valid_papers
        cites_in_corpus = [c for c in cites if c in paper_ids]
        if cites_in_corpus:
            queries.append((p.get('external_id', p.get('id')), build_paper_text(p), cites_in_corpus))
            
    if len(queries) > sample_size:
        queries = random.sample(queries, sample_size)
        
    logging.info(f"Encoding corpus of {len(paper_texts)} papers...")
    corpus_embs = model.encode(paper_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    
    mrr = 0.0
    precision = {k: 0.0 for k in k_values}
    
    for query_id, query_text, true_citations in queries:
        query_emb = model.encode(query_text, convert_to_tensor=True)
        # Compute similarities with all papers
        cos_scores = util.cos_sim(query_emb, corpus_embs)[0]
        
        # Sort scores descending
        top_results = np.argsort(-cos_scores.cpu().numpy())
        
        # Remove the query itself from results
        try:
            query_idx = paper_ids.index(query_id)
            top_results = top_results[top_results != query_idx]
        except ValueError:
            pass
            
        # Get ranked IDs
        ranked_ids = [paper_ids[idx] for idx in top_results]
        
        # Calculate MRR
        rank = 0
        for i, doc_id in enumerate(ranked_ids):
            if doc_id in true_citations:
                rank = i + 1
                break
        if rank > 0:
            mrr += 1.0 / rank
            
        # Calculate Precision@K
        for k in k_values:
            hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in true_citations)
            precision[k] += hits / k
            
    n_queries = len(queries) if queries else 1
    metrics = {
        'mrr': mrr / n_queries
    }
    for k in k_values:
        metrics[f'precision@{k}'] = precision[k] / n_queries
        
    return metrics

def evaluate_and_compare(finetuned_path: str = 'models/scibert-finetuned-papers', 
                         base_model_name: str = 'allenai/scibert_scivocab_uncased', 
                         triplets_path: str = 'data/triplets.csv', 
                         db_path: str = 'data/papers.db') -> Dict[str, Any]:
    """
    Evaluate and compare the fine-tuned model against the base model.
    """
    logging.info("Starting evaluation and comparison")
    
    # Load test triplets (last 20%)
    df = pd.read_csv(triplets_path)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    test_triplets = [tuple(x) for x in test_df.to_numpy()]
    
    # Load papers for retrieval eval
    papers = load_papers(db_path)
    
    results = {}
    
    from .train import create_model
    
    # Evaluate models
    for model_name, model_path in [("Base Model", base_model_name), ("Fine-tuned Model", finetuned_path)]:
        logging.info(f"--- Evaluating {model_name} ---")
        try:
            if os.path.exists(model_path):
                model = SentenceTransformer(model_path)
            else:
                model = create_model(model_path)
        except Exception as e:
            logging.error(f"Could not load model {model_path}: {e}")
            continue
            
        acc = compute_triplet_accuracy(model, test_triplets)
        retrieval_metrics = compute_mrr_and_precision(model, papers, k_values=[5, 10], sample_size=100)
        
        results[model_name] = {
            'triplet_accuracy': acc,
            **retrieval_metrics
        }
        
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model vs base model")
    parser.add_argument('--model', type=str, default='models/scibert-finetuned-papers', help='Path to fine-tuned model')
    parser.add_argument('--base-model', type=str, default='allenai/scibert_scivocab_uncased', help='Base model name')
    
    args = parser.parse_args()
    
    results = evaluate_and_compare(finetuned_path=args.model, base_model_name=args.base_model)
    
    print("\n" + "="*50)
    print("                EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<20} | {'Base Model':<12} | {'Fine-tuned Model'}")
    print("-" * 50)
    
    if "Base Model" in results and "Fine-tuned Model" in results:
        metrics = ['triplet_accuracy', 'mrr', 'precision@5', 'precision@10']
        for metric in metrics:
            base_val = results["Base Model"].get(metric, 0.0)
            ft_val = results["Fine-tuned Model"].get(metric, 0.0)
            print(f"{metric:<20} | {base_val:<12.4f} | {ft_val:.4f}")
    else:
        print("Evaluation incomplete. Missing model results.")
    print("="*50)
