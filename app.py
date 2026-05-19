"""
========================================================================
  INTELLIGENT WEB CONTENT SUMMARIZATION SYSTEM FOR ENHANCED ACCESSIBILITY
  -----------------------------------------------------------------------
  Author  : Abdurrahman Abubakar Said  |  CSA/2023/30777
  Dept    : Computer Science, Federal University Dutsin-Ma
  Session : 2024/2025
  Supervisor: Dr. Umar Iliyasu
========================================================================

HOW TO RUN
----------
1. Install dependencies:
       pip install streamlit requests beautifulsoup4 transformers torch
                   gtts nltk sumy rouge-score

2. Launch the app:
       streamlit run app.py

3. Open your browser at http://localhost:8501
========================================================================
"""

# ── Standard library ─────────────────────────────────────────────────
import os
import re
import sqlite3
import hashlib
import datetime
import io
import logging
import textwrap
from pathlib import Path
from typing import Tuple, List, Optional

# ── Third-party ───────────────────────────────────────────────────────
import streamlit as st
import requests
from bs4 import BeautifulSoup

# NLP / summarization
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Abstractive summarization (HuggingFace)
from transformers import pipeline, AutoTokenizer

# Text-to-Speech
from gtts import gTTS

# ── NLTK data (auto-download if missing) ─────────────────────────────
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1 – PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SummarizeAI – Web Content Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #21262d;
    --border:      #30363d;
    --accent:      #238636;
    --accent-soft: #1a7f37;
    --accent-glow: rgba(35,134,54,0.25);
    --amber:       #d29922;
    --red:         #da3633;
    --text:        #e6edf3;
    --muted:       #8b949e;
    --radius:      10px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1300px; }

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin: 0;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    margin: 0;
    color: var(--muted);
    font-size: 0.9rem;
}
.badge {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.04em;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

/* ── Summary box ── */
.summary-box {
    background: linear-gradient(135deg, #161b22, #0d1117);
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--text);
    margin: 1rem 0;
    box-shadow: 0 0 24px var(--accent-glow);
}

/* ── Metric pill ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 110px;
}
.metric-pill .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-pill .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Extracted text area ── */
.extracted-text {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--muted);
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
}

/* ── History table ── */
.hist-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 0;
    font-size: 0.88rem;
}
.hist-time { color: var(--muted); min-width: 140px; font-size: 0.78rem; }
.hist-url  { color: #58a6ff; word-break: break-all; flex: 1; }
.hist-mode { background: var(--surface2); border-radius: 5px; padding: 1px 7px; font-size: 0.75rem; }

/* ── Streamlit overrides ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    background: var(--accent-soft) !important;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stSlider"] {
    color: var(--text) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface) !important;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
}
.stAlert { border-radius: var(--radius) !important; }
div[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SECTION 2 – DATABASE LAYER (SQLite)
# ─────────────────────────────────────────────────────────────────────

DB_PATH = Path("summarize_ai.db")

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT    NOT NULL,
                mode        TEXT    NOT NULL,
                word_count  INTEGER,
                orig_words  INTEGER,
                summary     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER REFERENCES summaries(id),
                rating     INTEGER,
                comment    TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
        """)

def save_summary(url: str, mode: str, summary: str,
                 orig_words: int, sum_words: int) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO summaries (url,mode,word_count,orig_words,summary) "
            "VALUES (?,?,?,?,?)",
            (url, mode, sum_words, orig_words, summary),
        )
        return cur.lastrowid

def get_history(limit: int = 50) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def save_feedback(summary_id: int, rating: int, comment: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (summary_id,rating,comment) VALUES (?,?,?)",
            (summary_id, rating, comment),
        )

init_db()

# ─────────────────────────────────────────────────────────────────────
# SECTION 3 – WEB SCRAPER MODULE
# ─────────────────────────────────────────────────────────────────────

NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "noscript", "iframe", "figure", "figcaption",
    "button", "input", "select", "textarea", "svg", "canvas",
]
CONTENT_TAGS = ["article", "main", "section", "div", "p"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def scrape_url(url: str, timeout: int = 15) -> Tuple[str, str]:
    """
    Fetch a URL and return (page_title, clean_text).
    Raises ValueError on HTTP errors or empty content.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Page title
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    # Remove noise
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Try article/main first, fall back to body
    body = soup.find("article") or soup.find("main") or soup.find("body")
    if body is None:
        raise ValueError("Could not extract readable content from this page.")

    raw = body.get_text(separator=" ")

    # Clean whitespace
    text = re.sub(r"\s+", " ", raw).strip()
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)  # strip non-ASCII noise

    if len(text.split()) < 30:
        raise ValueError(
            "Very little text was found on this page. "
            "It may be JavaScript-rendered or behind a paywall."
        )

    return page_title, text

# ─────────────────────────────────────────────────────────────────────
# SECTION 4 – SUMMARIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────

# ── 4a. Extractive summarization (TextRank-inspired) ─────────────────

def _clean_summary(text: str) -> str:
    """Fix common T5 artefacts, capitalisation, and spacing."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])\1+", r"\1", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = " ".join(sentences)
    return text[0].upper() + text[1:] if text else text


def _sentence_scores(sentences: List[str], stop_words: set) -> dict:
    """Score sentences by word-frequency (simplified TextRank)."""
    freq: dict = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word.isalpha() and word not in stop_words:
                freq[word] = freq.get(word, 0) + 1

    if not freq:
        return {s: 0 for s in sentences}

    max_freq = max(freq.values())
    freq = {w: v / max_freq for w, v in freq.items()}

    scores: dict = {}
    for sent in sentences:
        score = 0.0
        for word in word_tokenize(sent.lower()):
            if word in freq:
                score += freq[word]
        scores[sent] = score
    return scores


def extractive_summarize(text: str, ratio: float = 0.3,
                          max_sentences: int = 10) -> str:
    """Return a summary by selecting top-scored sentences."""
    sentences = sent_tokenize(text)
    if len(sentences) <= 3:
        return text

    stop_words = set(stopwords.words("english"))
    scores = _sentence_scores(sentences, stop_words)

    n = max(2, min(max_sentences, int(len(sentences) * ratio)))
    top = sorted(scores, key=scores.get, reverse=True)[:n]

    # Preserve original sentence order
    ordered = [s for s in sentences if s in top]
    return _clean_summary(" ".join(ordered))


# ── 4b. Abstractive summarization (HuggingFace T5-small) ─────────────

@st.cache_resource(show_spinner=False)
def _load_abstractive_pipeline():
    """Load and cache the T5-small summarization pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    pipe = pipeline(
        "summarization",
        model="t5-small",
        tokenizer=tokenizer,
        framework="pt",
    )
    return pipe, tokenizer


def abstractive_summarize(text: str,
                           max_length: int = 180,
                           min_length: int = 60) -> str:
    """Generate an abstractive summary using T5-small."""
    pipe, tokenizer = _load_abstractive_pipeline()

    # T5-small has a 512-token limit — chunk if needed
    tokens = tokenizer.encode(text, truncation=False)
    CHUNK = 450  # tokens per chunk (leave room for output)

    if len(tokens) <= CHUNK:
        chunks = [text]
    else:
        # Split text into ~CHUNK-token chunks at sentence boundaries
        sentences = sent_tokenize(text)
        chunks, current, current_len = [], [], 0
        for sent in sentences:
            sent_len = len(tokenizer.encode(sent))
            if current_len + sent_len > CHUNK and current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))

    summaries = []
    per_max = max(60, max_length // max(len(chunks), 1))
    per_min = max(20, min_length // max(len(chunks), 1))

    for chunk in chunks:
        if len(chunk.split()) < 20:
            summaries.append(chunk)
            continue
        result = pipe(
            chunk,
            max_length=per_max,
            min_length=per_min,
            do_sample=False,
            truncation=True,
        )
        summaries.append(result[0]["summary_text"])

    final = " ".join(summaries)
    final = _clean_summary(final)
    return final


def summarize(text: str, mode: str,
              extractive_ratio: float = 0.3,
              abs_max: int = 180,
              abs_min: int = 60) -> str:
    """Dispatch to the appropriate summarizer."""
    if mode == "Extractive (TextRank)":
        return extractive_summarize(text, ratio=extractive_ratio)
    elif mode == "Abstractive (T5 AI)":
        return abstractive_summarize(text, max_length=abs_max, min_length=abs_min)
    else:  # Hybrid
        ext = extractive_summarize(text, ratio=0.5, max_sentences=15)
        return abstractive_summarize(ext, max_length=abs_max, min_length=abs_min)


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 – TEXT-TO-SPEECH MODULE (gTTS)
# ─────────────────────────────────────────────────────────────────────

LANG_MAP = {
    "English":  "en",
    "French":   "fr",
    "Spanish":  "es",
    "German":   "de",
    "Arabic":   "ar",
    "Hausa":    "ha",
}


def text_to_speech(text: str, lang_code: str = "en", slow: bool = False) -> bytes:
    """Convert text to MP3 bytes using gTTS."""
    # Ensure sentences are properly separated for natural pacing
    text = re.sub(r"([.!?])\s*", r"\1 ", text).strip()
    tts = gTTS(text=text, lang=lang_code, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────
# SECTION 6 – EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────

def compression_ratio(original: str, summary: str) -> float:
    orig_w = len(original.split())
    sum_w  = len(summary.split())
    if orig_w == 0:
        return 0.0
    return round((1 - sum_w / orig_w) * 100, 1)


def avg_sentence_length(text: str) -> float:
    sents = sent_tokenize(text)
    if not sents:
        return 0.0
    return round(sum(len(s.split()) for s in sents) / len(sents), 1)


def readability_label(avg_len: float) -> str:
    """Simple readability heuristic."""
    if avg_len < 12:
        return "🟢 Easy"
    elif avg_len < 20:
        return "🟡 Moderate"
    else:
        return "🔴 Complex"


# ─────────────────────────────────────────────────────────────────────
# SECTION 7 – SIDEBAR
# ─────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Syne;font-weight:800;color:#58a6ff;"
            "font-size:1.2rem;margin-bottom:0.2rem'>⚙️ Settings</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#8b949e;font-size:0.8rem;margin-top:0'>"
            "Tune the summarization parameters below.</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        mode = st.selectbox(
            "Summarization Mode",
            ["Extractive (TextRank)", "Abstractive (T5 AI)", "Hybrid (Extract → Abstract)"],
            help=(
                "Extractive: picks key sentences from the original.\n"
                "Abstractive: generates new sentences using AI (T5-small).\n"
                "Hybrid: extracts first, then abstracts — best quality."
            ),
        )

        st.markdown("**Length / Detail**")
        if mode == "Extractive (TextRank)":
            ext_ratio = st.slider(
                "Sentence keep ratio", 0.1, 0.6, 0.3, 0.05,
                help="Fraction of sentences to include in the summary.",
            )
            abs_max = abs_min = None
        else:
            ext_ratio = None
            abs_max = st.slider("Max summary length (tokens)", 80, 300, 180, 10)
            abs_min = st.slider("Min summary length (tokens)", 20, 100,  60, 10)

        st.divider()
        st.markdown("**♿ Accessibility**")
        tts_enabled = st.checkbox("Enable Text-to-Speech", value=True)
        tts_lang    = st.selectbox("TTS Language", list(LANG_MAP.keys()))
        tts_slow    = st.checkbox("Slow speech (easier to follow)", value=False)
        show_orig   = st.checkbox("Show extracted text", value=True)

        st.divider()
        st.markdown("**🌐 Scraping**")
        timeout = st.slider("Request timeout (s)", 5, 30, 15)

        st.divider()
        st.markdown(
            "<p style='font-size:0.72rem;color:#8b949e;line-height:1.6'>"
            "Built for <strong style='color:#e6edf3'>CSA/2023/30777</strong><br>"
            "Federal University Dutsin-Ma<br>"
            "Supervisor: Dr. Umar Iliyasu</p>",
            unsafe_allow_html=True,
        )

    return dict(
        mode=mode,
        ext_ratio=ext_ratio,
        abs_max=abs_max,
        abs_min=abs_min,
        tts_enabled=tts_enabled,
        tts_lang=tts_lang,
        tts_slow=tts_slow,
        show_orig=show_orig,
        timeout=timeout,
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION 8 – APP HEADER
# ─────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        """
        <div class="app-header">
          <div>
            <h1>📰 SummarizeAI</h1>
            <p>Intelligent Web Content Summarization for Enhanced Accessibility
               &nbsp;·&nbsp;
               <span class="badge">CSA/2023/30777</span>
               <span class="badge">Fed. Uni. Dutsin-Ma</span>
               <span class="badge">WCAG-Compliant</span>
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION 9 – SUMMARIZE TAB
# ─────────────────────────────────────────────────────────────────────

def render_summarize_tab(cfg: dict):
    st.markdown(
        "<div class='card-title'>Enter a URL to summarize</div>",
        unsafe_allow_html=True,
    )

    col_url, col_btn = st.columns([5, 1], gap="small")
    with col_url:
        url = st.text_input(
            "URL",
            placeholder="https://en.wikipedia.org/wiki/Natural_language_processing",
            label_visibility="collapsed",
        )
    with col_btn:
        go = st.button("Summarize →", use_container_width=True)

    # Also allow direct text input
    with st.expander("📋  Or paste text directly (optional)", expanded=False):
        direct_text = st.text_area(
            "Paste article text here",
            placeholder="Paste any article or document text here...",
            height=180,
            label_visibility="collapsed",
        )
        go_text = st.button("Summarize Text →")

    # ── Process URL ──────────────────────────────────────────────────
    if go and url.strip():
        _run_pipeline(
            source=url.strip(),
            source_type="url",
            cfg=cfg,
        )

    # ── Process direct text ──────────────────────────────────────────
    if go_text and direct_text.strip():
        _run_pipeline(
            source=direct_text.strip(),
            source_type="text",
            cfg=cfg,
        )

    # ── Show cached last result ──────────────────────────────────────
    if "last_result" in st.session_state and not go and not go_text:
        _render_result(st.session_state["last_result"], cfg)


def _run_pipeline(source: str, source_type: str, cfg: dict):
    """Full pipeline: scrape → summarize → TTS → store → display."""

    with st.status("Working…", expanded=True) as status:

        # 1. Content extraction
        if source_type == "url":
            st.write("🌐 Fetching and cleaning web page…")
            try:
                page_title, text = scrape_url(source, timeout=cfg["timeout"])
            except requests.exceptions.Timeout:
                status.update(label="⛔ Timed out", state="error")
                st.error("The request timed out. Try increasing the timeout in Settings.")
                return
            except requests.exceptions.HTTPError as e:
                status.update(label="⛔ HTTP error", state="error")
                st.error(f"HTTP error: {e}")
                return
            except Exception as e:
                status.update(label="⛔ Extraction failed", state="error")
                st.error(f"Could not extract content: {e}")
                return
        else:
            page_title = "Pasted Text"
            text = source

        orig_word_count = len(text.split())
        st.write(f"✅ Extracted **{orig_word_count:,}** words.")

        # 2. Summarization
        mode = cfg["mode"]
        st.write(f"🤖 Running **{mode}** summarization…")
        try:
            summary = summarize(
                text,
                mode=mode,
                extractive_ratio=cfg["ext_ratio"] or 0.3,
                abs_max=cfg["abs_max"] or 180,
                abs_min=cfg["abs_min"] or 60,
            )
        except Exception as e:
            status.update(label="⛔ Summarization failed", state="error")
            st.error(f"Summarization error: {e}")
            return

        sum_word_count = len(summary.split())
        st.write(f"✅ Summary: **{sum_word_count}** words.")

        # 3. TTS
        audio_bytes = None
        if cfg["tts_enabled"]:
            st.write("🔊 Generating audio…")
            try:
                lang_code = LANG_MAP[cfg["tts_lang"]]
                audio_bytes = text_to_speech(summary, lang_code, cfg["tts_slow"])
                st.write("✅ Audio ready.")
            except Exception as e:
                st.warning(f"TTS failed: {e}")

        # 4. Persist to DB
        try:
            rec_id = save_summary(
                url=source if source_type == "url" else "(direct text)",
                mode=mode,
                summary=summary,
                orig_words=orig_word_count,
                sum_words=sum_word_count,
            )
        except Exception:
            rec_id = None

        status.update(label="✅ Done!", state="complete", expanded=False)

    # 5. Store result in session and render
    result = dict(
        page_title=page_title,
        source=source,
        source_type=source_type,
        text=text,
        summary=summary,
        mode=mode,
        orig_word_count=orig_word_count,
        sum_word_count=sum_word_count,
        audio_bytes=audio_bytes,
        rec_id=rec_id,
    )
    st.session_state["last_result"] = result
    _render_result(result, cfg)


def _render_result(result: dict, cfg: dict):
    """Render the summarization result cards."""

    st.markdown("---")

    # ── Page title ───────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='font-family:Syne;font-weight:700;color:#e6edf3'>"
        f"📄 {result['page_title']}</h3>",
        unsafe_allow_html=True,
    )

    # ── Metrics row ──────────────────────────────────────────────────
    cr   = compression_ratio(result["text"], result["summary"])
    asl  = avg_sentence_length(result["summary"])
    rdlb = readability_label(asl)
    n_sents = len(sent_tokenize(result["summary"]))

    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-pill"><div class="val">{result['orig_word_count']:,}</div><div class="lbl">Original words</div></div>
          <div class="metric-pill"><div class="val">{result['sum_word_count']}</div><div class="lbl">Summary words</div></div>
          <div class="metric-pill"><div class="val">{cr}%</div><div class="lbl">Compression</div></div>
          <div class="metric-pill"><div class="val">{n_sents}</div><div class="lbl">Sentences</div></div>
          <div class="metric-pill"><div class="val">{asl}</div><div class="lbl">Avg sent. len</div></div>
          <div class="metric-pill"><div class="val" style="font-size:1rem">{rdlb}</div><div class="lbl">Readability</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary ──────────────────────────────────────────────────────
    st.markdown(
        f"<div class='card-title' style='margin-top:1.2rem'>📝 Summary — "
        f"<span style='color:#3fb950'>{result['mode']}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='summary-box'>{result['summary']}</div>",
        unsafe_allow_html=True,
    )

    # ── Download buttons ─────────────────────────────────────────────
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button(
            "📥 Download Summary (.txt)",
            data=result["summary"],
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dc2:
        full_report = (
            f"URL: {result['source']}\n"
            f"Mode: {result['mode']}\n"
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Original words: {result['orig_word_count']}\n"
            f"Summary words:  {result['sum_word_count']}\n"
            f"Compression:    {cr}%\n\n"
            f"--- SUMMARY ---\n\n{result['summary']}\n\n"
            f"--- ORIGINAL TEXT (first 3000 chars) ---\n\n{result['text'][:3000]}"
        )
        st.download_button(
            "📄 Download Full Report (.txt)",
            data=full_report,
            file_name="full_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dc3:
        if result.get("audio_bytes"):
            st.download_button(
                "🎵 Download Audio (.mp3)",
                data=result["audio_bytes"],
                file_name="summary.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )

    # ── Audio player ─────────────────────────────────────────────────
    if cfg["tts_enabled"] and result.get("audio_bytes"):
        st.markdown(
            "<div class='card-title' style='margin-top:1rem'>🔊 Audio Summary</div>",
            unsafe_allow_html=True,
        )
        st.audio(result["audio_bytes"], format="audio/mp3")
        st.caption(
            f"Language: {cfg['tts_lang']} · "
            f"Speed: {'Slow' if cfg['tts_slow'] else 'Normal'} · "
            "Powered by Google Text-to-Speech"
        )

    # ── Extracted text ───────────────────────────────────────────────
    if cfg["show_orig"]:
        with st.expander("📜 Extracted original text", expanded=False):
            st.markdown(
                f"<div class='extracted-text'>{result['text'][:4000]}"
                f"{'…' if len(result['text']) > 4000 else ''}</div>",
                unsafe_allow_html=True,
            )

    # ── Feedback ─────────────────────────────────────────────────────
    if result.get("rec_id"):
        with st.expander("⭐ Rate this summary", expanded=False):
            rating  = st.select_slider(
                "Quality", options=[1, 2, 3, 4, 5], value=4,
                format_func=lambda x: "⭐" * x,
            )
            comment = st.text_input("Optional comment")
            if st.button("Submit feedback"):
                save_feedback(result["rec_id"], rating, comment)
                st.success("Thanks for your feedback!")


# ─────────────────────────────────────────────────────────────────────
# SECTION 10 – HISTORY TAB
# ─────────────────────────────────────────────────────────────────────

def render_history_tab():
    st.markdown(
        "<div class='card-title'>Recent Summarization Sessions</div>",
        unsafe_allow_html=True,
    )
    rows = get_history(limit=50)
    if not rows:
        st.info("No summaries yet. Head to the **Summarize** tab to get started.")
        return

    # Stats
    total_orig = sum(r["orig_words"] or 0 for r in rows)
    total_sum  = sum(r["word_count"]  or 0 for r in rows)
    overall_cr = round((1 - total_sum / total_orig) * 100, 1) if total_orig else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total sessions", len(rows))
    sc2.metric("Words processed", f"{total_orig:,}")
    sc3.metric("Avg compression", f"{overall_cr}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    for r in rows:
        url_display = r["url"][:80] + ("…" if len(r["url"]) > 80 else "")
        st.markdown(
            f"""
            <div class="hist-row">
              <div class="hist-time">{r['created_at']}</div>
              <div class="hist-url">{url_display}</div>
              <div><span class="hist-mode">{r['mode']}</span></div>
              <div style="color:#8b949e;min-width:90px;text-align:right">
                {r['orig_words']:,} → {r['word_count']} w
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Expand a summary from history"):
        ids   = [r["id"]  for r in rows]
        label = [f"#{r['id']} — {r['url'][:60]}…  ({r['created_at']})" for r in rows]
        sel   = st.selectbox("Select session", label)
        if sel:
            idx  = label.index(sel)
            row  = rows[idx]
            st.markdown(
                f"<div class='summary-box'>{row['summary']}</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download",
                data=row["summary"],
                file_name=f"summary_{row['id']}.txt",
            )


# ─────────────────────────────────────────────────────────────────────
# SECTION 11 – ABOUT TAB
# ─────────────────────────────────────────────────────────────────────

def render_about_tab():
    st.markdown(
        """
## About This System

**Development of an Intelligent Web Content Summarization System for Enhanced Accessibility**
is a final-year Computer Science project submitted to the **Department of Computer Science,
Federal University Dutsin-Ma**, Katsina State.

---

### 🔬 System Architecture

| Module | Technology |
|--------|-----------|
| Frontend / UI | Streamlit (Python) |
| Web Scraping | Requests + BeautifulSoup 4 |
| Extractive NLP | NLTK – TextRank-inspired scoring |
| Abstractive AI | HuggingFace Transformers – T5-small |
| Text-to-Speech | Google Text-to-Speech (gTTS) |
| Database | SQLite (via Python `sqlite3`) |
| Accessibility | WCAG-compliant layout, TTS, adjustable output length |

---

### 🧪 Evaluation Metrics

- **Compression Ratio** — fraction of original content retained  
- **Average Sentence Length** — proxy for readability  
- **ROUGE / BLEU** — semantic accuracy (offline evaluation)  
- **WCAG Compliance** — screen-reader compatible, keyboard-navigable  

---

### 👤 Project Details

| Field | Detail |
|-------|--------|
| Student | Abdurrahman Abubakar Said |
| Matric No. | CSA/2023/30777 |
| Department | Computer Science |
| University | Federal University Dutsin-Ma |
| Supervisor | Dr. Umar Iliyasu |
| Session | 2024/2025 |

---

### 📦 Dependencies

```
streamlit>=1.34
requests>=2.31
beautifulsoup4>=4.12
transformers>=4.40
torch>=2.0
nltk>=3.8
gTTS>=2.5
```

Install with:
```bash
pip install streamlit requests beautifulsoup4 transformers torch nltk gTTS
```
        """,
        unsafe_allow_html=False,
    )


# ─────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    render_header()

    cfg = render_sidebar()

    tab_sum, tab_hist, tab_about = st.tabs(
        ["📰  Summarize", "🕑  History", "ℹ️  About"]
    )

    with tab_sum:
        render_summarize_tab(cfg)

    with tab_hist:
        render_history_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
