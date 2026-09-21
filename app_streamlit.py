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
from src.downloader import download_paper_aria2, find_aria2_executable
from src.tts_engine import (
    get_available_voices,
    synthesize_speech_wav,
    speak_live_background,
    stop_live_speech
)

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

    /* Paper Reader Panel */
    .reader-panel {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 24px 28px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    .reader-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.4;
        margin-bottom: 10px;
    }
    .reader-meta {
        font-size: 0.83rem;
        color: #64748B;
        margin-bottom: 14px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 10px;
    }
    .reader-abstract-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .reader-abstract {
        font-size: 0.97rem;
        color: #1E293B;
        line-height: 1.75;
        text-align: justify;
    }
    .tts-bar {
        display: flex;
        gap: 10px;
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid #E2E8F0;
        align-items: center;
    }
    .tts-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 600;
        cursor: pointer;
        border: none;
        transition: all 0.15s ease;
    }
    .tts-read {
        background: #2563EB;
        color: white;
    }
    .tts-read:hover {
        background: #1D4ED8;
    }
    .tts-stop {
        background: #EF4444;
        color: white;
    }
    .tts-stop:hover {
        background: #DC2626;
    }
    .tts-status {
        font-size: 0.82rem;
        color: #64748B;
        margin-left: 4px;
    }
    .pdf-frame-container {
        margin-top: 18px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading SciBERT recommendation model...")
def get_recommender():
    """Load and cache the recommender engine."""
    from src.recommender.recommend import PaperRecommender
    return PaperRecommender()


def make_tts_html(paper_id: str, text: str) -> str:
    """
    Generate a self-contained HTML block with Read / Stop buttons
    that use the browser's Web Speech API to read the given text.

    Args:
        paper_id: Unique identifier string for this paper (used as element IDs).
        text: The text that should be spoken aloud.

    Returns:
        HTML string to be rendered via st.components.v1.html.
    """
    import json
    text = str(text or "")
    js_text = json.dumps(text).replace("</script", "<\\/script").replace("</Script", "<\\/Script")
    html = f"""
    <div class="tts-bar" id="tts-bar-{paper_id}">
        <button class="tts-btn tts-read" id="read-btn-{paper_id}"
            onclick="startReading_{paper_id}()">
            &#9654; Read
        </button>
        <button class="tts-btn tts-stop" id="stop-btn-{paper_id}"
            onclick="stopReading_{paper_id}()"
            style="display:none;">
            &#9632; Stop
        </button>
        <span class="tts-status" id="tts-status-{paper_id}"></span>
    </div>
    <script>
        var _synth_{paper_id} = window.speechSynthesis;
        var _utt_{paper_id} = null;

        function startReading_{paper_id}() {{
            if (!window.speechSynthesis) {{
                document.getElementById('tts-status-{paper_id}').textContent = '❌ TTS not supported in this browser.';
                return;
            }}
            _synth_{paper_id}.cancel();
            _utt_{paper_id} = new SpeechSynthesisUtterance({js_text});
            window._currentUtterance = _utt_{paper_id};
            _utt_{paper_id}.rate = 0.95;
            _utt_{paper_id}.pitch = 1.0;
            _utt_{paper_id}.onstart = function() {{
                document.getElementById('tts-status-{paper_id}').textContent = '🔊 Reading...';
                document.getElementById('read-btn-{paper_id}').style.display = 'none';
                document.getElementById('stop-btn-{paper_id}').style.display = 'inline-flex';
            }};
            _utt_{paper_id}.onend = function() {{
                document.getElementById('tts-status-{paper_id}').textContent = '✅ Done.';
                document.getElementById('read-btn-{paper_id}').style.display = 'inline-flex';
                document.getElementById('stop-btn-{paper_id}').style.display = 'none';
            }};
            _utt_{paper_id}.onerror = function(e) {{
                document.getElementById('tts-status-{paper_id}').textContent = '❌ Error: ' + e.error;
                document.getElementById('read-btn-{paper_id}').style.display = 'inline-flex';
                document.getElementById('stop-btn-{paper_id}').style.display = 'none';
            }};
            _synth_{paper_id}.speak(_utt_{paper_id});
        }}

        function stopReading_{paper_id}() {{
            _synth_{paper_id}.cancel();
            document.getElementById('tts-status-{paper_id}').textContent = '⏹ Stopped.';
            document.getElementById('read-btn-{paper_id}').style.display = 'inline-flex';
            document.getElementById('stop-btn-{paper_id}').style.display = 'none';
        }}
    </script>
    <style>
        .tts-bar {{ display:flex; gap:10px; margin-top:12px; align-items:center; }}
        .tts-btn {{ display:inline-flex; align-items:center; gap:6px; padding:8px 18px;
                   border-radius:6px; font-size:0.88rem; font-weight:600; cursor:pointer;
                   border:none; transition:all 0.15s ease; font-family:inherit; }}
        .tts-read {{ background:#2563EB; color:white; }}
        .tts-read:hover {{ background:#1D4ED8; }}
        .tts-stop {{ background:#EF4444; color:white; }}
        .tts-stop:hover {{ background:#DC2626; }}
        .tts-status {{ font-size:0.82rem; color:#64748B; }}
    </style>
    """
    return html



@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_paper_text(arxiv_id: str) -> str:
    """
    Fetch the full text of an arXiv paper from ar5iv.org.

    Args:
        arxiv_id: Clean arXiv ID (e.g. '2304.00501').

    Returns:
        Extracted plain text of the paper, or empty string on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://ar5iv.org/html/{arxiv_id}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PaperReader/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "figure", "table", "aside"]):
            tag.decompose()

        article = soup.find("article") or soup.find("div", class_="ltx_document") or soup.body
        if not article:
            return ""

        paragraphs = []
        for elem in article.find_all(["p", "h1", "h2", "h3", "h4"]):
            text = elem.get_text(separator=" ", strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)
        return full_text if len(full_text) > 200 else ""
    except Exception:
        return ""


@st.cache_data(ttl=3600)
def get_s2_pdf_url(s2_paper_id: str) -> str:
    """
    Query the free Semantic Scholar API to get the open-access PDF URL.
    Returns the PDF URL or empty string if unavailable.
    """
    if not s2_paper_id or len(s2_paper_id) < 10:
        return ""
    try:
        import requests
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_paper_id}?fields=openAccessPdf,externalIds"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            oa = data.get("openAccessPdf") or {}
            if oa.get("url"):
                return oa["url"]
            ext_ids = data.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arxiv")
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception:
        pass
    return ""


@st.cache_data
def get_cached_voices():
    try:
        return get_available_voices()
    except Exception:
        return [{"id": "default", "name": "Default System Voice", "gender": "Neutral"}]


def render_paper_reader(r: dict, card_index: int, show_score: bool = True, score_class: str = "score-high"):
    """
    Render a paper card.
    Inside the expander:
      1. Abstract shown immediately.
      2. ⚡ Download PDF (aria2) button.
      3. After download, PDF text is extracted automatically.
      4. ▶ Read (local TTS) and ⏹ Stop buttons to listen to the full paper.
    """
    import streamlit.components.v1 as components
    from src.downloader import extract_text_from_pdf

    raw_title = r.get('title')
    title = str(raw_title) if raw_title and not pd.isna(raw_title) else 'Untitled Paper'
    year = r.get('year') or 'N/A'
    source = str(r.get('source') or 'arxiv').replace('_', ' ')
    external_id = str(r.get('external_id') or '')
    pdf_url = str(r.get('pdf_url') or '')

    raw_abs = r.get('abstract')
    abstract = str(raw_abs) if raw_abs and not pd.isna(raw_abs) else 'No abstract available.'
    score = r.get('score', 0.0)

    # Canonical URL
    if pdf_url:
        paper_url = pdf_url
    elif external_id:
        paper_url = f"https://arxiv.org/abs/{external_id}"
    else:
        paper_url = "#"

    is_arxiv = source.lower() == 'arxiv' or 'arxiv.org' in paper_url

    # Build PDF download target URL
    clean_eid = str(external_id).replace("v1", "").replace("v2", "").strip()
    is_s2 = source.lower() == "semantic_scholar"

    if is_arxiv and clean_eid and not is_s2:
        # Direct arXiv paper
        pdf_target = f"https://arxiv.org/pdf/{clean_eid}.pdf"
    elif is_s2 and clean_eid:
        # Look up the real PDF link from Semantic Scholar API (cached)
        pdf_target = get_s2_pdf_url(clean_eid)
    elif pdf_url and pdf_url.endswith(".pdf"):
        pdf_target = pdf_url
    else:
        pdf_target = ""

    # Unique keys for this paper
    uid = f"{card_index}_{abs(hash(title)) % 100000}"
    dl_key   = f"dl_{uid}"      # stores (ok, path, size)
    text_key = f"txt_{uid}"     # stores extracted full text string
    wav_key  = f"wav_{uid}"     # stores synthesized audio bytes

    # Score badge
    score_pct = f"{score * 100:.1f}%"
    score_html = f'<span class="score-badge {score_class}">Score: {score_pct}</span>' if show_score else ''

    # ---- Paper card ----
    st.markdown(f"""
    <div class="paper-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="flex:1; padding-right:16px;">
                <div style="margin-bottom:6px;">
                    <span class="source-badge">{source}</span>
                    <span class="meta-tag">Year: {year}</span>
                    <span class="meta-tag">ID: {external_id or 'N/A'}</span>
                </div>
                <h4 style="margin:0 0 6px 0; font-size:1.05rem;">
                    <a href="{paper_url}" target="_blank" style="color:#0F172A; text-decoration:none;">
                        {title}
                    </a>
                </h4>
            </div>
            <div>{score_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Open Paper", expanded=False):
        # Abstract
        st.markdown(f"""
        <div class="reader-panel">
            <div class="reader-abstract-label">Abstract</div>
            <div class="reader-abstract">{abstract}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Step 1: Download PDF ──────────────────────────────────────────
        dl_state = st.session_state.get(dl_key)

        if dl_state is None:
            # Not yet downloaded — show Download button
            if pdf_target:
                col_btn, col_link = st.columns([1, 2])
                with col_btn:
                    if st.button("⚡ Download PDF (aria2)", key=f"dl_btn_{uid}"):
                        with st.spinner("Downloading via aria2…"):
                            safe_name = f"{clean_eid or abs(hash(title)) % 100000}.pdf".replace("/", "_")
                            ok, path, sz = download_paper_aria2(pdf_target, filename=safe_name)
                        if ok:
                            st.session_state[dl_key] = ("ok", path, sz)
                            with st.spinner("Extracting text from PDF…"):
                                extracted = extract_text_from_pdf(path)
                            st.session_state[text_key] = extracted
                        else:
                            st.session_state[dl_key] = ("error", path, 0)
                        st.rerun()
                with col_link:
                    if is_s2:
                        s2_page = f"https://www.semanticscholar.org/paper/{clean_eid}"
                        st.markdown(
                            f'<a href="{s2_page}" target="_blank" '
                            f'style="font-size:0.85rem; color:#7C3AED;">🔗 Open on Semantic Scholar</a>',
                            unsafe_allow_html=True
                        )
                    elif is_arxiv and clean_eid:
                        st.markdown(
                            f'<a href="https://arxiv.org/abs/{clean_eid}" target="_blank" '
                            f'style="font-size:0.85rem; color:#2563EB;">🔗 Open on arXiv</a>',
                            unsafe_allow_html=True
                        )
            else:
                # No open-access PDF found
                page_link = (
                    f"https://www.semanticscholar.org/paper/{clean_eid}" if is_s2
                    else (paper_url if paper_url != "#" else "")
                )
                if page_link:
                    st.markdown(
                        f'ℹ️ No open-access PDF found. '
                        f'<a href="{page_link}" target="_blank" style="color:#6366F1;">Open paper page ↗</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No PDF available for this paper.")


        elif dl_state[0] == "error":
            st.error(f"Download failed: {dl_state[1]}")
            if st.button("Retry Download", key=f"retry_{uid}"):
                del st.session_state[dl_key]
                if text_key in st.session_state:
                    del st.session_state[text_key]
                st.rerun()

        else:
            # ── Download succeeded ─────────────────────────────────────────
            _, pdf_path, sz = dl_state
            sz_mb = sz / (1024 * 1024)
            st.success(f"✅ PDF downloaded ({sz_mb:.2f} MB)")

            # Save-to-disk button
            col_save, col_redo = st.columns([2, 1])
            with col_save:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"💾 Save PDF ({sz_mb:.2f} MB)",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        key=f"save_{uid}"
                    )
            with col_redo:
                if st.button("🗑 Remove", key=f"rm_{uid}"):
                    del st.session_state[dl_key]
                    if text_key in st.session_state:
                        del st.session_state[text_key]
                    if wav_key in st.session_state:
                        del st.session_state[wav_key]
                    st.rerun()

            st.markdown("---")

            # ── Step 2: Extracted text ─────────────────────────────────────
            extracted_text = st.session_state.get(text_key, "")
            if not extracted_text:
                # Try to extract if not done yet
                with st.spinner("Extracting text from PDF…"):
                    extracted_text = extract_text_from_pdf(pdf_path)
                st.session_state[text_key] = extracted_text

            if extracted_text:
                word_count = len(extracted_text.split())
                with st.expander(f"📄 Full Paper Text ({word_count:,} words)", expanded=False):
                    st.text_area(
                        label="",
                        value=extracted_text[:15000] + ("\n\n[...truncated for display...]" if len(extracted_text) > 15000 else ""),
                        height=400,
                        key=f"txt_area_{uid}",
                        disabled=True
                    )
            else:
                st.warning("Could not extract text from this PDF (may be scanned/image-based).")
                extracted_text = abstract  # fall back to abstract

            # ── Step 3: Read / Stop ────────────────────────────────────────
            st.markdown("**🎧 Listen to the paper:**")

            voices = get_cached_voices()
            voice_labels = [f"{v['name']} ({v['gender']})" for v in voices]

            col_v, col_r = st.columns([3, 2])
            with col_v:
                sel_idx = st.selectbox(
                    "Voice",
                    range(len(voices)),
                    format_func=lambda i: voice_labels[i],
                    key=f"vsel_{uid}"
                )
                chosen_voice_id = voices[sel_idx]["id"]
            with col_r:
                speaking_rate = st.slider(
                    "Speed (wpm)",
                    min_value=120, max_value=260, value=175, step=5,
                    key=f"spd_{uid}"
                )

            col_read, col_stop = st.columns([1, 1])
            with col_read:
                if st.button("▶ Read Full Paper", key=f"read_{uid}"):
                    with st.spinner("Synthesizing audio… this may take a moment for long papers."):
                        text_for_tts = f"{title}. {extracted_text}"
                        ok, wav_path_out, wav_bytes = synthesize_speech_wav(
                            text=text_for_tts,
                            voice_id=chosen_voice_id,
                            rate=speaking_rate,
                            max_chars=20000
                        )
                    if ok:
                        st.session_state[wav_key] = wav_bytes
                    else:
                        st.error(f"Audio synthesis failed: {wav_path_out}")

            with col_stop:
                if st.button("⏹ Stop", key=f"stop_{uid}"):
                    stop_live_speech()
                    if wav_key in st.session_state:
                        del st.session_state[wav_key]

            # Audio player
            if st.session_state.get(wav_key):
                wav_bytes = st.session_state[wav_key]
                st.audio(wav_bytes, format="audio/wav")
                st.download_button(
                    label="💾 Download Audio (.wav)",
                    data=wav_bytes,
                    file_name=f"{clean_eid or 'paper'}_narration.wav",
                    mime="audio/wav",
                    key=f"dl_wav_{uid}"
                )


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
        if "search_results" not in st.session_state:
            st.session_state["search_results"] = None
        if "search_query" not in st.session_state:
            st.session_state["search_query"] = ""
        if "search_info" not in st.session_state:
            st.session_state["search_info"] = None
        if "search_terms_str" not in st.session_state:
            st.session_state["search_terms_str"] = None

        query_input = st.text_input(
            "Enter Research Topic or Query:",
            value=st.session_state.get("search_query", ""),
            placeholder="e.g. GNN in drug discovery, RLHF in LLMs, protein function prediction...",
            help="Type any scientific domain or research question."
        )
        
        col_s_btn, col_s_clear = st.columns([4, 1])
        with col_s_btn:
            search_clicked = st.button("Search Recommendations", type="primary", use_container_width=True)
        with col_s_clear:
            if st.button("Clear Results", key="clear_topic_search", use_container_width=True):
                st.session_state["search_results"] = None
                st.session_state["search_info"] = None
                st.session_state["search_terms_str"] = None
                st.session_state["search_query"] = ""
                st.rerun()

        if search_clicked and query_input:
            st.session_state["search_query"] = query_input
            terms_str = None
            if enable_expansion:
                expansion_data = recommender.expander.expand(query_input)
                if expansion_data.get("all_expansion_terms"):
                    terms_str = " · ".join(f"`{t}`" for t in expansion_data["all_expansion_terms"])
            st.session_state["search_terms_str"] = terms_str

            with st.spinner(f"Retrieving recommendations for '{query_input}'..."):
                if mode == "Local Index Only":
                    results = recommender.recommend_by_topic(query_input, top_k=top_k, use_expansion=enable_expansion)
                    st.session_state["search_info"] = ("info", "Served from local FAISS vector index.")
                elif mode == "Force Live Scrape":
                    results = recommender.recommend_adaptive(
                        query=query_input,
                        top_k=top_k,
                        similarity_threshold=0.99,
                        auto_train=auto_train,
                        use_lora=use_qlora,
                        max_scrape=max_scrape
                    )
                    st.session_state["search_info"] = ("success", "Live scraped papers from online sources and updated the index.")
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
                        st.session_state["search_info"] = ("info", "High similarity match found in local corpus. Served from index.")
                    else:
                        st.session_state["search_info"] = ("success", "Information was sparse in local corpus. Scraped fresh papers and updated index with Q-LoRA fine-tuning.")

            st.session_state["search_results"] = results

        # Render persisted search results
        if st.session_state.get("search_results") is not None:
            results = st.session_state["search_results"]
            terms_str = st.session_state.get("search_terms_str")
            if terms_str:
                st.markdown(f"<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem;'><span style='color: #475569; font-weight: 600;'>Expanded Concepts:</span> {terms_str}</div>", unsafe_allow_html=True)

            if st.session_state.get("search_info"):
                kind, msg = st.session_state["search_info"]
                if kind == "info":
                    st.info(msg)
                elif kind == "success":
                    st.success(msg)

            if not results:
                st.warning("No matching papers found. Try broadening your query.")
            else:
                st.subheader(f"Top {len(results)} Recommendations")
                for i, r in enumerate(results):
                    score = r.get('score', 0.0)
                    if score >= 0.55:
                        badge_class = "score-high"
                    elif score >= 0.40:
                        badge_class = "score-mid"
                    else:
                        badge_class = "score-low"
                    render_paper_reader(r, card_index=i, show_score=True, score_class=badge_class)

    elif mode == "Seed Paper Similarity":
        st.subheader("Find Papers Similar to a Reference Paper")
        if "seed_results" not in st.session_state:
            st.session_state["seed_results"] = None
        if "seed_title_saved" not in st.session_state:
            st.session_state["seed_title_saved"] = ""

        seed_title = st.text_input("Paper Title:", placeholder="e.g. Attention Is All You Need")
        seed_abstract = st.text_area("Paper Abstract:", placeholder="Paste the abstract of the reference paper...", height=150)
        
        col_seed_btn, col_seed_clear = st.columns([4, 1])
        with col_seed_btn:
            seed_clicked = st.button("Find Similar Papers", type="primary", use_container_width=True)
        with col_seed_clear:
            if st.button("Clear Results", key="clear_seed_search", use_container_width=True):
                st.session_state["seed_results"] = None
                st.session_state["seed_title_saved"] = ""
                st.rerun()

        if seed_clicked:
            if not seed_title or not seed_abstract:
                st.error("Please provide both a title and abstract.")
            else:
                with st.spinner("Calculating similarity embeddings..."):
                    results = recommender.recommend_by_paper(seed_title, seed_abstract, top_k=top_k)
                    st.session_state["seed_results"] = results
                    st.session_state["seed_title_saved"] = seed_title

        # Render persisted seed similarity results
        if st.session_state.get("seed_results") is not None:
            results = st.session_state["seed_results"]
            if not results:
                st.warning("No similar papers found.")
            else:
                st.subheader(f"Top {len(results)} Papers Similar to '{st.session_state['seed_title_saved']}'")
                for i, r in enumerate(results):
                    render_paper_reader(r, card_index=1000 + i, show_score=True, score_class="score-high")

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
