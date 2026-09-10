from dotenv import load_dotenv
load_dotenv()

import sys
import os
import tempfile

# Ensure stdout/stderr handle unicode properly in Streamlit context
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import time
from utils.audio_processor import process_input, DOWNLOAD_DIR
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoMind AI — Meeting Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ══════════════════════════════════════════
   ROOT VARIABLES
══════════════════════════════════════════ */
:root {
    --bg:           #030712;
    --bg-2:         #060b18;
    --glass:        rgba(255,255,255,0.04);
    --glass-hover:  rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.10);
    --glass-border-hover: rgba(255,255,255,0.22);
    --neon-purple:  #a855f7;
    --neon-cyan:    #22d3ee;
    --neon-pink:    #f472b6;
    --neon-green:   #4ade80;
    --neon-yellow:  #facc15;
    --text:         #f1f5f9;
    --text-muted:   #64748b;
    --text-dim:     #94a3b8;
    --shadow-purple: rgba(168,85,247,0.35);
    --shadow-cyan:   rgba(34,211,238,0.25);
}

/* ══════════════════════════════════════════
   GLOBAL RESET & BASE
══════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
    min-height: 100vh;
}

/* ── Animated Gradient Background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(168,85,247,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 90% 80%, rgba(34,211,238,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(244,114,182,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Subtle grid */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(168,85,247,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(168,85,247,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(3, 7, 18, 0.85) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stSidebarNav"] { display: none; }

/* ══════════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero ── */
.hero-wrap {
    padding: 2.5rem 0 1.5rem;
    position: relative;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--neon-purple);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-purple), transparent);
    border-radius: 2px;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.08;
    margin: 0 0 1rem;
    background: linear-gradient(135deg,
        #ffffff 0%,
        var(--neon-purple) 40%,
        var(--neon-cyan) 75%,
        var(--neon-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 40px rgba(168,85,247,0.3));
}

.hero-sub {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: var(--text-dim);
    line-height: 1.6;
    max-width: 520px;
}

/* Sidebar hero */
.sidebar-brand {
    padding: 0.5rem 0 1rem;
}

.sidebar-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--neon-purple) 60%, var(--neon-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.sidebar-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--text-muted);
    text-transform: uppercase;
}

/* ══════════════════════════════════════════
   GLASS CARDS
══════════════════════════════════════════ */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    border-color: var(--glass-border-hover);
    transform: translateY(-2px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(168,85,247,0.08);
}

/* Gradient top stripe */
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan), var(--neon-pink));
    opacity: 0;
    transition: opacity 0.3s;
}

.glass-card:hover::before {
    opacity: 1;
}

/* Glow blob inside card */
.glass-card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.card-title-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 8px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    font-size: 0.85rem;
    flex-shrink: 0;
}

.card-content {
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--text-dim);
}

/* Title card variant */
.title-card {
    background: linear-gradient(135deg,
        rgba(168,85,247,0.12) 0%,
        rgba(34,211,238,0.06) 50%,
        rgba(244,114,182,0.06) 100%);
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}

.title-card-text {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.35;
}

.title-card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--neon-purple);
    margin-bottom: 0.6rem;
}

/* ══════════════════════════════════════════
   BADGES
══════════════════════════════════════════ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 99px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: all 0.2s;
}

.badge-purple {
    background: rgba(168,85,247,0.15);
    color: var(--neon-purple);
    border: 1px solid rgba(168,85,247,0.3);
    box-shadow: 0 0 12px rgba(168,85,247,0.1);
}

.badge-cyan {
    background: rgba(34,211,238,0.1);
    color: var(--neon-cyan);
    border: 1px solid rgba(34,211,238,0.25);
    box-shadow: 0 0 12px rgba(34,211,238,0.08);
}

.badge-green {
    background: rgba(74,222,128,0.1);
    color: var(--neon-green);
    border: 1px solid rgba(74,222,128,0.25);
    box-shadow: 0 0 12px rgba(74,222,128,0.08);
}

.badge-pink {
    background: rgba(244,114,182,0.1);
    color: var(--neon-pink);
    border: 1px solid rgba(244,114,182,0.25);
    box-shadow: 0 0 12px rgba(244,114,182,0.08);
}

/* ══════════════════════════════════════════
   INPUTS & FORM ELEMENTS
══════════════════════════════════════════ */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.25s !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.12), 0 0 20px rgba(168,85,247,0.1) !important;
    background: rgba(168,85,247,0.05) !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-dim) !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--neon-purple) 0%, #7c3aed 50%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.7rem 1.75rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: none !important;
    box-shadow: 0 4px 24px rgba(168,85,247,0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 12px 40px rgba(168,85,247,0.5), 0 0 60px rgba(168,85,247,0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: none !important;
    color: var(--text-dim) !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--glass-border-hover) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-dim) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.25s !important;
    box-shadow: none !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(168,85,247,0.4) !important;
    color: var(--text) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(168,85,247,0.2) !important;
}

/* ══════════════════════════════════════════
   PIPELINE STATUS BARS
══════════════════════════════════════════ */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.7rem 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    margin: 0.35rem 0;
    border: 1px solid rgba(255,255,255,0.07);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    color: var(--text-dim);
    transition: all 0.25s;
}

.status-bar:hover {
    background: rgba(255,255,255,0.05);
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    transition: all 0.3s;
}

.dot-active {
    background: var(--neon-purple);
    box-shadow: 0 0 10px var(--neon-purple), 0 0 20px rgba(168,85,247,0.5);
    animation: neon-pulse 1.2s ease-in-out infinite;
}

.dot-done {
    background: var(--neon-green);
    box-shadow: 0 0 8px var(--neon-green);
}

.dot-pending {
    background: rgba(255,255,255,0.1);
}

@keyframes neon-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ══════════════════════════════════════════
   TRANSCRIPT BOX
══════════════════════════════════════════ */
.transcript-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.9;
    max-height: 320px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ══════════════════════════════════════════
   CHAT UI
══════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    transition: border-color 0.2s !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(168,85,247,0.25) !important;
}

[data-testid="stChatInputContainer"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px) !important;
}

/* Chat section header */
.chat-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.chat-header-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(34,211,238,0.15));
    border: 1px solid rgba(168,85,247,0.3);
    font-size: 1.2rem;
}

.chat-header-text {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--text), var(--neon-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Empty chat state */
.empty-chat {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    text-align: center;
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 18px;
    margin-bottom: 1rem;
}

.empty-chat-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px rgba(168,85,247,0.6));
}

.empty-chat-text {
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.7;
    max-width: 280px;
}

/* ══════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════ */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
    min-height: 55vh;
}

.empty-state-orb {
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%,
        rgba(168,85,247,0.5) 0%,
        rgba(34,211,238,0.3) 50%,
        rgba(244,114,182,0.2) 100%);
    margin-bottom: 2rem;
    filter: blur(0px);
    box-shadow:
        0 0 40px rgba(168,85,247,0.4),
        0 0 80px rgba(168,85,247,0.2),
        inset 0 0 30px rgba(255,255,255,0.1);
    animation: orb-float 4s ease-in-out infinite;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
}

@keyframes orb-float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    33% { transform: translateY(-12px) rotate(2deg); }
    66% { transform: translateY(-6px) rotate(-2deg); }
}

.empty-state-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.75rem;
}

.empty-state-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    max-width: 400px;
    line-height: 1.75;
    margin-bottom: 2rem;
}

.feature-badges {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    justify-content: center;
}

/* ══════════════════════════════════════════
   STREAMLIT OVERRIDES
══════════════════════════════════════════ */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan)) !important;
    border-radius: 4px !important;
}

.stSpinner > div {
    border-top-color: var(--neon-purple) !important;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--text-dim) !important;
    line-height: 1.75 !important;
}

label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* Success/Error/Info messages */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(168,85,247,0.4) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* Divider */
.stDivider { opacity: 0.15 !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(168,85,247,0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(168,85,247,0.6);
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🎬 VideoMind</div>
        <div class="sidebar-tagline">AI · Meeting Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<span class="badge badge-purple">⚡ Input Source</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Input mode selector
    input_mode = st.radio(
        "Input Mode",
        ["🔗  YouTube URL / File Path", "📁  Upload Audio/Video File"],
        index=0,
        label_visibility="collapsed",
    )

    source = ""
    uploaded_file = None

    if input_mode == "🔗  YouTube URL / File Path":
        source = st.text_input(
            "URL or Path",
            placeholder="https://youtube.com/watch?v=... or path/to/file.mp4",
        )
    else:
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["mp4", "mp3", "wav", "m4a", "webm", "ogg", "avi", "mov", "mkv"],
        )
        if uploaded_file is not None:
            st.success(f"✅ Ready: {uploaded_file.name}")

    st.markdown("<br>", unsafe_allow_html=True)
    language = st.selectbox("🌐 Language", ["english", "hinglish"], index=0)
    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button("⚡  Analyse Now", use_container_width=True)

    if st.session_state.pipeline_done or st.session_state.processing:
        st.divider()
        st.markdown('<span class="badge badge-green">📊 Pipeline Status</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">AI-Powered · Real-Time Analysis</div>
    <div class="hero-title">VideoMind AI</div>
    <div class="hero-sub">
        Transform any video or audio into structured insights — transcripts, summaries, action items, and an interactive AI chat interface.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    # Validate input
    effective_source = None
    if uploaded_file is not None:
        # Save uploaded file to downloads directory
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        save_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        effective_source = save_path
    elif source.strip():
        effective_source = source.strip().strip('"').strip("'")

    if not effective_source:
        st.error("Please enter a YouTube URL, file path, or upload a file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.processing = True
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_status(key, state, message):
            st.session_state.pipeline_steps[key] = state
            if state == "active":
                progress_placeholder.info(f"⚙️ {message}")

        try:
            update_status("audio", "active", "Step 1/6: Downloading and processing audio...")
            chunks = process_input(effective_source)
            st.session_state.pipeline_steps["audio"] = "done"

            update_status("transcript", "active", "Step 2/6: Transcribing audio with AI (Speech-to-Text)...")
            transcript = transcribe_all(chunks, language)
            st.session_state.pipeline_steps["transcript"] = "done"

            update_status("title", "active", "Step 3/6: Generating session title...")
            title = generate_title(transcript)
            st.session_state.pipeline_steps["title"] = "done"

            update_status("summary", "active", "Step 4/6: Creating executive summary...")
            summary = summarize(transcript)
            st.session_state.pipeline_steps["summary"] = "done"

            update_status("extract", "active", "Step 5/6: Extracting action items, decisions & questions...")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            st.session_state.pipeline_steps["extract"] = "done"

            update_status("rag", "active", "Step 6/6: Building RAG vector store for interactive chat...")
            rag_chain = build_rag_chain(transcript)
            st.session_state.pipeline_steps["rag"] = "done"

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.session_state.processing = False
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            st.session_state.processing = False
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="title-card">
        <div class="title-card-label">📌 Session Title</div>
        <div class="title-card-text">{r['title']}</div>
    </div>""", unsafe_allow_html=True)

    # Export buttons
    export_col1, export_col2, _ = st.columns([1, 1, 4], gap="small")
    with export_col1:
        export_text = f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n## Action Items\n{r['action_items']}\n\n## Key Decisions\n{r['key_decisions']}\n\n## Open Questions\n{r['open_questions']}\n\n## Full Transcript\n{r['transcript']}"
        st.download_button(
            "📄 Export .md",
            data=export_text.encode("utf-8"),
            file_name="meeting_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "📋 Export .txt",
            data=export_text.encode("utf-8"),
            file_name="meeting_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                <span class="card-title-icon">📋</span>
                Executive Summary
            </div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                <span class="card-title-icon">✅</span>
                Action Items
            </div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                <span class="card-title-icon">🔑</span>
                Key Decisions
            </div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                <span class="card-title-icon">❓</span>
                Open Questions
            </div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-icon">💬</div>
        <div class="chat-header-text">Chat with your Meeting</div>
        <span class="badge badge-cyan" style="margin-left:auto">RAG · Powered</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display using native Streamlit chat components
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.markdown("""
        <div class="empty-chat">
            <div class="empty-chat-icon">✨</div>
            <div class="empty-chat-text">Ask anything about your meeting — key points, follow-ups, who said what, and more.</div>
        </div>""", unsafe_allow_html=True)

    # Modern chat input — submits on Enter, auto-clears
    user_input = st.chat_input("Ask a question about your meeting...")
    if user_input and user_input.strip():
        with st.spinner("Thinking..."):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-orb">🎬</div>
        <div class="empty-state-title">Ready to Analyse</div>
        <div class="empty-state-sub">
            Paste a YouTube URL or local file path in the sidebar,
            or upload an audio/video file directly.
            Choose your language and hit <strong>Analyse Now</strong> to unlock insights.
        </div>
        <div class="feature-badges">
            <span class="badge badge-purple">🎤 Transcription</span>
            <span class="badge badge-cyan">📋 Summarisation</span>
            <span class="badge badge-green">✅ Action Items</span>
            <span class="badge badge-pink">🔑 Key Decisions</span>
            <span class="badge badge-purple">🧠 RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)