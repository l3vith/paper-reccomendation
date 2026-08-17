"""
Streamlit Web Application for Research Paper Recommendation System.

Features:
  - Clean, minimal, modern UI.
  - Adaptive Search (Sparsity Detection + Live Web Fallback).
  - Paper-to-Paper Similarity mode (Title & Abstract input).
  - Corpus Knowledge Base Explorer with search & filtering.
  - Benchmark Metrics & Model Evaluation dashboard.
"""

import os
import json
import sqlite3
import warnings
import logging
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Research Paper Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .header-container {
        padding: 8px 0 20px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #E2E8F0;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Paper Card Styling */
    .paper-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .paper-card:hover {
        border-color: #94A3B8;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
    }
    
    /* Score Badge */
    .score-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .score-high {
        background-color: #DCFCE7;
        color: #166534;
    }
    .score-mid {
        background-color: #FEF9C3;
        color: #854D0E;
    }
    .score-low {
        background-color: #F1F5F9;
        color: #334155;
    }
    
    .source-badge {
        background-color: #F1F5F9;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
    }
    
    .meta-tag {
        color: #64748B;
        font-size: 0.82rem;
        margin-right: 12px;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading SciBERT recommendation model...")
def get_recommender():
    """Load and cache the recommender engine."""
    from src.recommender.recommend import PaperRecommender
    return PaperRecommender()


def get_db_stats():
    """Retrieve corpus statistics from SQLite and FAISS."""
    db_path = "data/papers.db"
    total_papers = 0
    sources = {}
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM papers")
            total_papers = c.fetchone()[0]
            c.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
            sources = dict(c.fetchall())
            conn.close()
        except Exception:
            pass

    faiss_vectors = 0
    if os.path.exists("data/faiss_index.bin"):
        try:
            import faiss
            index = faiss.read_index("data/faiss_index.bin")
            faiss_vectors = index.ntotal
        except Exception:
            pass

    return total_papers, sources, faiss_vectors


# Clean Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Research Paper Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    mode = st.radio(
        "Search Mode",
        options=["Adaptive Search (Smart)", "Seed Paper Similarity", "Local Index Only", "Force Live Scrape"],
        index=0,
        help="Adaptive Search checks the local FAISS index first and only scrapes + fine-tunes if results are sparse."
    )
    
    st.divider()
    
    top_k = st.slider("Top Recommendations (K)", min_value=3, max_value=20, value=6, step=1)
    
    with st.expander("Advanced Configuration", expanded=False):
        enable_expansion = st.checkbox("Enable Query Expansion", value=True, help="Expands query with scientific acronyms and domain concepts.")
        use_qlora = st.checkbox("Use Q-LoRA (4-bit NF4)", value=True, help="Applies 4-bit NormalFloat4 Quantization with LoRA adapters for maximum parameter and memory efficiency.")
        threshold = st.slider(
            "Sparsity Threshold",
            min_value=0.20,
            max_value=0.70,
            value=0.40,
            step=0.05,
            help="Cosine similarity score below which the system considers information sparse and triggers live scraping."
        )
        auto_train = st.checkbox("Auto-train on Sparse Discovery", value=True, help="Fine-tunes SciBERT on newly mined triplets when new topics are fetched.")
        max_scrape = st.number_input("Max Papers to Scrape", min_value=10, max_value=200, value=40, step=10)

    st.divider()
    
    st.markdown("### Corpus Statistics")
    total_papers, sources, faiss_vectors = get_db_stats()
    
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Indexed Vectors", f"{faiss_vectors:,}")
    col_stat2.metric("Saved Papers", f"{total_papers:,}")
    
    if sources:
        st.caption("Sources: " + " | ".join([f"{k}: {v}" for k, v in sources.items()]))
        
    st.divider()
    
    if st.button("Rebuild Vector Index", width="stretch"):
        with st.spinner("Re-indexing SQLite corpus into FAISS..."):
            from src.recommender.indexer import PaperIndex
            idx = PaperIndex()
            c = idx.build_index()
            st.success(f"Successfully indexed {c} papers.")
            st.rerun()

# Main Tabs
tab_search, tab_eval, tab_corpus = st.tabs(["Paper Discovery", "Model Benchmarks", "Corpus Explorer"])

# --- TAB 1: Paper Discovery ---
with tab_search:
    recommender = get_recommender()
    
    if mode in ["Adaptive Search (Smart)", "Local Index Only", "Force Live Scrape"]:
        query_input = st.text_input(
            "Enter Research Topic or Query:",
            placeholder="e.g. GNN in drug discovery, RLHF in LLMs, protein function prediction...",
            help="Type any scientific domain or research question."
        )
        
        search_clicked = st.button("Search Recommendations", type="primary", width="stretch")

        if search_clicked and query_input:
            # Query Expansion Preview
            if enable_expansion:
                expansion_data = recommender.expander.expand(query_input)
                if expansion_data.get("all_expansion_terms"):
                    terms_str = " · ".join(f"`{t}`" for t in expansion_data["all_expansion_terms"])
                    st.markdown(f"<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem;'><span style='color: #475569; font-weight: 600;'>Expanded Concepts:</span> {terms_str}</div>", unsafe_allow_html=True)

            with st.spinner(f"Retrieving recommendations for '{query_input}'..."):
                if mode == "Local Index Only":
                    results = recommender.recommend_by_topic(query_input, top_k=top_k, use_expansion=enable_expansion)
                    st.info("Served from local FAISS vector index.")
                elif mode == "Force Live Scrape":
                    results = recommender.recommend_adaptive(
                        query=query_input,
                        top_k=top_k,
                        similarity_threshold=0.99,
                        auto_train=auto_train,
                        use_lora=use_qlora,
                        max_scrape=max_scrape
                    )
                    st.success("Live scraped papers from online sources and updated the index.")
                else:  # Adaptive Search (Smart)
                    pre_check = recommender.recommend_by_topic(query_input, top_k=top_k, use_expansion=enable_expansion)
                    has_good_match = bool(pre_check and pre_check[0].get('score', 0) >= threshold)
                    
                    results = recommender.recommend_adaptive(
                        query=query_input,
                        top_k=top_k,
                        similarity_threshold=threshold,
                        auto_train=auto_train,
                        use_lora=use_qlora,
                        max_scrape=max_scrape
                    )
                    
                    if has_good_match:
                        st.info("High similarity match found in local corpus. Served from index.")
                    else:
                        st.success("Information was sparse in local corpus. Scraped fresh papers and updated index with Q-LoRA fine-tuning.")

            # Display Results
            if not results:
                st.warning("No matching papers found. Try broadening your query.")
            else:
                st.subheader(f"Top {len(results)} Recommendations")
                
                for r in results:
                    score = r.get('score', 0.0)
                    score_pct = f"{score * 100:.1f}%"
                    
                    if score >= 0.55:
                        badge_class = "score-high"
                    elif score >= 0.40:
                        badge_class = "score-mid"
                    else:
                        badge_class = "score-low"

                    title = r.get('title', 'Untitled Paper')
                    year = r.get('year') or 'N/A'
                    source = (r.get('source') or 'arxiv').replace('_', ' ')
                    url = r.get('pdf_url') or f"https://arxiv.org/abs/{r.get('external_id')}"
                    abstract = r.get('abstract', 'No abstract available.')

                    st.markdown(f"""
                    <div class="paper-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="flex: 1; padding-right: 16px;">
                                <div style="margin-bottom: 6px;">
                                    <span class="source-badge">{source}</span>
                                    <span class="meta-tag">Year: {year}</span>
                                    <span class="meta-tag">ID: {r.get('external_id', 'N/A')}</span>
                                </div>
                                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem;">
                                    <a href="{url}" target="_blank" style="color: #0F172A; text-decoration: none;">
                                        {title}
                                    </a>
                                </h4>
                            </div>
                            <div>
                                <span class="score-badge {badge_class}">Score: {score_pct}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Abstract and Details"):
                        st.write(abstract)
                        st.markdown(f"[Open Full Paper Link]({url})")

    elif mode == "Seed Paper Similarity":
        st.subheader("Find Papers Similar to a Reference Paper")
        seed_title = st.text_input("Paper Title:", placeholder="e.g. Attention Is All You Need")
        seed_abstract = st.text_area("Paper Abstract:", placeholder="Paste the abstract of the reference paper...", height=150)
        
        if st.button("Find Similar Papers", type="primary", width="stretch"):
            if not seed_title or not seed_abstract:
                st.error("Please provide both a title and abstract.")
            else:
                with st.spinner("Calculating similarity embeddings..."):
                    results = recommender.recommend_by_paper(seed_title, seed_abstract, top_k=top_k)
                    
                if not results:
                    st.warning("No similar papers found.")
                else:
                    st.subheader(f"Top {len(results)} Papers Similar to '{seed_title}'")
                    for r in results:
                        score = r.get('score', 0.0)
                        url = r.get('pdf_url') or f"https://arxiv.org/abs/{r.get('external_id')}"
                        st.markdown(f"""
                        <div class="paper-card">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <span class="source-badge">{r.get('source', 'arXiv')}</span>
                                    <span class="meta-tag">Year: {r.get('year', 'N/A')}</span>
                                    <h4 style="margin: 6px 0;"><a href="{url}" target="_blank" style="text-decoration: none; color: #0F172A;">{r.get('title')}</a></h4>
                                </div>
                                <div>
                                    <span class="score-badge score-high">Score: {score:.4f}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("Abstract"):
                            st.write(r.get('abstract', ''))
                            st.markdown(f"[Open Full Paper Link]({url})")

# --- TAB 2: Model Benchmarks ---
with tab_eval:
    st.subheader("Model Evaluation and Benchmarks")
    st.caption("Quantitative comparison of Fine-Tuned SciBERT (Q-LoRA 4-bit) vs. Base SciBERT across scientific embedding tasks.")
    
    st.info("Q-LoRA Configuration: 4-bit NormalFloat4 (NF4) Quantization with Double Quantization | Rank r=16, Alpha=32, Target: query, value, key, dense | Trainable Parameters: 2,678,784 / 112,597,248 (2.38%)")
    
    eval_path = "results/evaluation_metrics.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)
            
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        
        ft_acc = eval_data.get("Fine-tuned Model", {}).get("triplet_accuracy", 0.0)
        base_acc = eval_data.get("Base Model", {}).get("triplet_accuracy", 0.0)
        col_b1.metric("Triplet Accuracy", f"{ft_acc * 100:.1f}%", delta=f"{(ft_acc - base_acc)*100:+.1f}% vs Base")
        
        ft_mrr = eval_data.get("Fine-tuned Model", {}).get("mrr", 0.0)
        base_mrr = eval_data.get("Base Model", {}).get("mrr", 0.0)
        col_b2.metric("Mean Reciprocal Rank", f"{ft_mrr:.3f}", delta=f"{(ft_mrr - base_mrr):+.3f}")
        
        ft_p5 = eval_data.get("Fine-tuned Model", {}).get("precision@5", 0.0)
        base_p5 = eval_data.get("Base Model", {}).get("precision@5", 0.0)
        col_b3.metric("Precision @ 5", f"{ft_p5 * 100:.1f}%", delta=f"{(ft_p5 - base_p5)*100:+.1f}%")
        
        ft_p10 = eval_data.get("Fine-tuned Model", {}).get("precision@10", 0.0)
        base_p10 = eval_data.get("Base Model", {}).get("precision@10", 0.0)
        col_b4.metric("Precision @ 10", f"{ft_p10 * 100:.1f}%", delta=f"{(ft_p10 - base_p10)*100:+.1f}%")

        st.subheader("Metric Comparison")
        comparison_rows = []
        for metric in ['triplet_accuracy', 'mrr', 'precision@5', 'precision@10']:
            b_val = eval_data.get("Base Model", {}).get(metric, 0.0)
            f_val = eval_data.get("Fine-tuned Model", {}).get(metric, 0.0)
            diff = f_val - b_val
            pct_gain = (diff / b_val * 100) if b_val > 0 else 0
            comparison_rows.append({
                "Metric": metric.replace('_', ' ').title(),
                "Base SciBERT": f"{b_val:.4f}",
                "Fine-Tuned SciBERT": f"{f_val:.4f}",
                "Absolute Gain": f"{diff:+.4f}",
                "Relative Improvement": f"{pct_gain:+.1f}%"
            })
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch")
    else:
        st.info("Evaluation metrics have not been computed yet.")
        
        if st.button("Run Evaluation Benchmark"):
            with st.spinner("Evaluating models on test triplets and citation ranking..."):
                from src.model.evaluate import evaluate_and_compare
                evaluate_and_compare()
                st.success("Evaluation complete.")
                st.rerun()

# --- TAB 3: Corpus Explorer ---
with tab_corpus:
    st.subheader("Local Knowledge Base")
    db_path = "data/papers.db"
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df_papers = pd.read_sql("SELECT id, source, external_id, title, year, pdf_url FROM papers ORDER BY id DESC", conn)
        conn.close()
        
        search_kw = st.text_input("Filter saved corpus by title keyword:", "")
        if search_kw:
            df_filtered = df_papers[df_papers['title'].str.contains(search_kw, case=False, na=False)]
        else:
            df_filtered = df_papers
            
        st.caption(f"Showing {len(df_filtered)} of {len(df_papers)} papers in SQLite database.")
        st.dataframe(
            df_filtered,
            column_config={
                "pdf_url": st.column_config.LinkColumn("Paper Link"),
                "year": st.column_config.NumberColumn("Year", format="%d")
            },
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("No papers database found at data/papers.db.")
