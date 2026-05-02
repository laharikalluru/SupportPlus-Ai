"""
SupportPlus AI — Premium Dark Chatbot UI.

Run from the project root:
    streamlit run ui.py

Requires the FastAPI backend:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
import uuid
import json
from datetime import datetime, date
from typing import Any, Dict, List

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL      = os.getenv("SUPPORTPLUS_API_URL", "http://localhost:8000/ask")
HEALTH_URL   = API_URL.replace("/ask", "/health")
DEFAULT_USER = os.getenv("SUPPORTPLUS_USER_ID", "user1")
HISTORY_FILE = os.path.join("app", "data", "ui_history.json")

_MODEL_LABELS = {
    "gemini":   ("gemini", "Gemini Flash"),
    "groq":     ("groq",   "Groq · Llama 3.1"),
    "fallback": ("fallback","FAQ + Web"),
    "unknown":  ("unknown", "Unknown"),
}

_MODEL_COLORS = {
    "gemini":   "#4f8ef7",
    "groq":     "#f59e0b",
    "fallback": "#10b981",
    "unknown":  "#64748b",
}

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────── */
* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.stDeployButton { display: none; }

/* ── Page background ─────────────────────────────────────────────────── */
.stApp {
    background: #0a0e1a !important;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(79,142,247,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(139,92,246,0.10) 0%, transparent 50%) !important;
}

/* ── Main container ──────────────────────────────────────────────────── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 5rem !important;
    max-width: 820px !important;
}

/* ── Header banner ───────────────────────────────────────────────────── */
.sp-topbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10,14,26,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0.85rem 1.2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.sp-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #4f8ef7, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(79,142,247,0.4);
}
.sp-title-wrap { flex: 1; }
.sp-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.01em;
}
.sp-status {
    font-size: 0.72rem;
    color: #10b981;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin-top: 1px;
}
.sp-status::before {
    content: '';
    width: 6px;
    height: 6px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}

/* ── Welcome screen ──────────────────────────────────────────────────── */
.sp-welcome {
    text-align: center;
    padding: 3rem 1rem 2rem;
    animation: fadeInUp 0.6s ease;
}
.sp-welcome-logo {
    width: 72px;
    height: 72px;
    background: linear-gradient(135deg, #4f8ef7, #8b5cf6);
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    margin: 0 auto 1.2rem;
    box-shadow: 0 8px 32px rgba(79,142,247,0.35);
}
.sp-welcome h1 {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 0.4rem;
    letter-spacing: -0.03em;
}
.sp-welcome p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0 0 2rem;
}
.sp-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    max-width: 560px;
    margin: 0 auto;
}
.sp-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.sp-chip:hover {
    background: rgba(79,142,247,0.1);
    border-color: rgba(79,142,247,0.3);
    color: #4f8ef7;
    transform: translateY(-1px);
}

/* ── Chat messages ───────────────────────────────────────────────────── */
.stChatMessage {
    animation: fadeInUp 0.35s ease !important;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User bubble */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(79,142,247,0.08) !important;
    border: 1px solid rgba(79,142,247,0.15) !important;
    border-radius: 18px 18px 4px 18px !important;
    margin-left: 3rem !important;
}

/* Assistant bubble */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px 18px 18px 4px !important;
    margin-right: 3rem !important;
}

/* Avatar styling */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #4f8ef7, #3b6fd4) !important;
    border-radius: 12px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
    border-radius: 12px !important;
}

/* Message text */
div[data-testid="stChatMessage"] p {
    color: #e2e8f0 !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

/* ── Model badge ─────────────────────────────────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.73rem;
    font-weight: 500;
    color: #64748b;
    margin-top: 0.5rem;
    padding: 0.2rem 0.7rem;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    transition: all 0.2s;
}
.badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.latency-pill {
    font-size: 0.7rem;
    color: #475569;
    margin-left: 0.3rem;
    padding: 0.1rem 0.4rem;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
}

/* ── Sources block ───────────────────────────────────────────────────── */
.sources-block {
    margin-top: 0.8rem;
    padding: 0.75rem 1rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #4f8ef7;
    border-radius: 0 12px 12px 0;
    font-size: 0.81rem;
}
.sources-title {
    font-weight: 600;
    color: #4f8ef7;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.source-faq { color: #94a3b8; padding: 0.18rem 0; display: flex; align-items: flex-start; gap: 0.4rem; }
.source-faq::before { content: '📄'; font-size: 0.8rem; flex-shrink: 0; margin-top: 1px; }
.source-web { padding: 0.18rem 0; display: flex; align-items: flex-start; gap: 0.4rem; }
.source-web::before { content: '🌐'; font-size: 0.8rem; flex-shrink: 0; margin-top: 1px; }
.source-web a { color: #60a5fa; text-decoration: none; }
.source-web a:hover { text-decoration: underline; color: #93c5fd; }

/* ── Chat input override ─────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(79,142,247,0.5) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important;
}
[data-testid="stChatInputTextArea"] {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: #475569 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

.sidebar-section {
    margin-bottom: 1.5rem;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.stat-label {
    font-size: 0.72rem;
    color: #475569;
    margin-bottom: 0.2rem;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
}
.model-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
    color: #94a3b8;
}
.model-row:last-child { border-bottom: none; }
.model-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.model-count {
    margin-left: auto;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Sidebar inputs ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] label { color: #64748b !important; font-size: 0.8rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(239,68,68,0.1) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #f87171 !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #4f8ef7 !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }

/* ── Conversation history list ────────────────────────────────── */
.conv-group-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.6rem 0 0.3rem;
}
.conv-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.7rem;
    border-radius: 10px;
    cursor: pointer;
    font-size: 0.82rem;
    color: #94a3b8;
    border: 1px solid transparent;
    margin-bottom: 2px;
    transition: all 0.18s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conv-item:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.08);
    color: #e2e8f0;
}
.conv-item.active {
    background: rgba(79,142,247,0.1);
    border-color: rgba(79,142,247,0.2);
    color: #93c5fd;
}
.conv-icon { font-size: 0.75rem; flex-shrink: 0; }
.new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.6rem;
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 10px;
    color: #60a5fa;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    margin-bottom: 1rem;
    transition: all 0.2s;
}
.new-chat-btn:hover {
    background: rgba(79,142,247,0.18);
    border-color: rgba(79,142,247,0.4);
    transform: translateY(-1px);
}
</style>
"""

# ---------------------------------------------------------------------------
# Session state & Persistence helpers
# ---------------------------------------------------------------------------

def _load_history() -> List[Dict]:
    """Load conversation history from local JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert string timestamps back to datetime
            for c in data:
                if "created_at" in c:
                    try:
                        c["created_at"] = datetime.fromisoformat(c["created_at"])
                    except Exception:
                        c["created_at"] = datetime.now()
            return data
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


def _save_history(convs: List[Dict]) -> None:
    """Save conversation history to local JSON file."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        # Create a copy to serialize created_at
        to_save = []
        for c in convs:
            c_copy = c.copy()
            if isinstance(c_copy["created_at"], datetime):
                c_copy["created_at"] = c_copy["created_at"].isoformat()
            to_save.append(c_copy)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def _make_conv() -> Dict:
    """Create a blank conversation dict."""
    return {
        "id":           str(uuid.uuid4())[:8],
        "title":        "New Chat",
        "messages":     [],
        "created_at":   datetime.now(),
        "total_queries": 0,
        "model_counts": {"gemini": 0, "groq": 0, "fallback": 0, "unknown": 0},
        "last_latency": None,
    }


def _init_session() -> None:
    if "conversations" not in st.session_state:
        loaded = _load_history()
        if loaded:
            st.session_state.conversations = loaded
            st.session_state.current_conv_id = loaded[0]["id"]
        else:
            first = _make_conv()
            st.session_state.conversations = [first]
            st.session_state.current_conv_id = first["id"]
            _save_history(st.session_state.conversations)

    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER


def _current_conv() -> Dict:
    """Return the active conversation dict."""
    cid = st.session_state.current_conv_id
    for c in st.session_state.conversations:
        if c["id"] == cid:
            return c
    # fallback: first conv
    return st.session_state.conversations[0]


def _switch_conv(conv_id: str) -> None:
    st.session_state.current_conv_id = conv_id
    st.rerun()


def _new_chat() -> None:
    conv = _make_conv()
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_conv_id = conv["id"]
    _save_history(st.session_state.conversations)
    st.rerun()

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(query: str, user_id: str) -> Dict[str, Any]:
    payload = {"query": query, "user_id": user_id, "top_k": 2, "web_results": 1}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(API_URL, json=payload)
        r.raise_for_status()
        return r.json()

# ---------------------------------------------------------------------------
# Typing animation
# ---------------------------------------------------------------------------

def _stream_text(placeholder: Any, text: str, delay: float = 0.016) -> None:
    words = text.split(" ")
    chunk_mode = len(text) > 600
    displayed = ""
    if chunk_mode:
        chunk: List[str] = []
        for word in words:
            chunk.append(word)
            if len(chunk) >= 8:
                displayed += " ".join(chunk) + " "
                placeholder.markdown(displayed + "▌")
                chunk = []
                time.sleep(delay * 4)
        if chunk:
            displayed += " ".join(chunk)
        placeholder.markdown(displayed)
    else:
        for word in words:
            displayed += word + " "
            placeholder.markdown(displayed + "▌")
            time.sleep(delay)
        placeholder.markdown(displayed.strip())

# ---------------------------------------------------------------------------
# Source renderer
# ---------------------------------------------------------------------------

def _render_sources(faq_sources: List[Dict], web_sources: List[Dict]) -> str:
    if not faq_sources and not web_sources:
        return ""
    lines = ['<div class="sources-block">']
    lines.append('<div class="sources-title">📎 Sources</div>')
    for f in faq_sources:
        q     = (f.get("question") or "FAQ entry").strip()
        score = f.get("score", 0.0)
        sec   = f.get("section", "")
        sec_s = f" <span style='color:#475569'>· {sec}</span>" if sec else ""
        lines.append(f'<div class="source-faq">{q}{sec_s} <span style="color:#334155">({score:.0%})</span></div>')
    for w in web_sources:
        title = (w.get("title") or "Web source").strip()
        url   = w.get("url", "#")
        lines.append(f'<div class="source-web"><a href="{url}" target="_blank">{title}</a></div>')
    lines.append("</div>")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Model badge
# ---------------------------------------------------------------------------

def _model_badge(model_key: str, latency_ms: float | None) -> str:
    _, label = _MODEL_LABELS.get(model_key, ("unknown", model_key.title()))
    color    = _MODEL_COLORS.get(model_key, "#64748b")
    lat      = f'<span class="latency-pill">⚡ {latency_ms:.0f}ms</span>' if latency_ms else ""
    return (
        f'<div class="model-badge">'
        f'<span class="badge-dot" style="background:{color}"></span>'
        f'{label}{lat}'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        # Logo + New Chat
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">'
            '<div style="width:32px;height:32px;background:linear-gradient(135deg,#4f8ef7,#8b5cf6);'
            'border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1rem;">🤖</div>'
            '<div style="font-weight:700;color:#f1f5f9;font-size:0.95rem;">SupportPlus AI</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("✏️  New Chat", use_container_width=True, key="btn_new_chat"):
            _new_chat()

        st.markdown("---")

        # ── Conversation history ───────────────────────────────────────────
        convs = st.session_state.conversations
        cid   = st.session_state.current_conv_id

        # Group by date
        today     = date.today()
        groups: Dict[str, List] = {"Today": [], "Yesterday": [], "Earlier": []}
        for c in convs:
            d = c["created_at"].date()
            delta = (today - d).days
            if delta == 0:
                groups["Today"].append(c)
            elif delta == 1:
                groups["Yesterday"].append(c)
            else:
                groups["Earlier"].append(c)

        for group_name, group_convs in groups.items():
            if not group_convs:
                continue
            st.markdown(f'<div class="conv-group-label">{group_name}</div>', unsafe_allow_html=True)
            for c in group_convs:
                is_active = c["id"] == cid
                active_cls = "active" if is_active else ""
                title = c["title"][:34] + "…" if len(c["title"]) > 34 else c["title"]
                # Render as button (Streamlit limitation — no true HTML click routing)
                btn_label = f"{'💬' if is_active else '🗨️'}  {title}"
                if st.button(btn_label, key=f"conv_{c['id']}", use_container_width=True):
                    _switch_conv(c["id"])

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)
        st.text_input("User ID", key="user_id", placeholder="e.g. user1")

        st.markdown("---")

        # ── Stats for current conv ─────────────────────────────────────────
        conv = _current_conv()
        st.markdown('<div class="sidebar-section-title">Session Stats</div>', unsafe_allow_html=True)
        total = conv["total_queries"]
        lat   = conv["last_latency"]
        lat_s = f"{lat:.0f}ms" if lat else "—"
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">Queries</div>'
                f'<div class="stat-value">{total}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">Latency</div>'
                f'<div class="stat-value">{lat_s}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Model usage ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">Model Usage</div>', unsafe_allow_html=True)
        mc = conv["model_counts"]
        for key, (_, label) in _MODEL_LABELS.items():
            if key == "unknown":
                continue
            color = _MODEL_COLORS[key]
            st.markdown(
                f'<div class="model-row">'
                f'<span class="model-dot" style="background:{color}"></span>'
                f'{label}<span class="model-count">{mc.get(key, 0)}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:0.4rem;">'
            '<a href="http://localhost:8000/docs" target="_blank" style="color:#60a5fa;font-size:0.82rem;text-decoration:none;">📖 API Docs</a>'
            '<a href="http://localhost:8000/health" target="_blank" style="color:#60a5fa;font-size:0.82rem;text-decoration:none;">💚 Health Check</a>'
            '</div>',
            unsafe_allow_html=True,
        )

    return (st.session_state.get("user_id") or DEFAULT_USER).strip() or DEFAULT_USER

# ---------------------------------------------------------------------------
# Welcome screen (shown when no messages yet)
# ---------------------------------------------------------------------------

_SUGGESTIONS = [
    "How do I reset my password?",
    "What are your support hours?",
    "How can I track my order?",
    "How do I cancel my subscription?",
]

def _render_welcome() -> None:
    st.markdown(
        '<div class="sp-welcome">'
        '<div class="sp-welcome-logo">🤖</div>'
        '<h1>SupportPlus AI</h1>'
        '<p>Your AI-powered support assistant — powered by Gemini &amp; Groq<br>'
        'with a live FAQ knowledge base and real-time web data.</p>'
        '<div class="sp-chips">'
        + "".join(f'<div class="sp-chip">{s}</div>' for s in _SUGGESTIONS)
        + '</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SupportPlus AI",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _init_session()
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Top bar ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sp-topbar">'
        '<div class="sp-logo">🤖</div>'
        '<div class="sp-title-wrap">'
        '<div class="sp-name">SupportPlus AI</div>'
        '<div class="sp-status">Online · Ready to help</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    uid = _render_sidebar()

    # ── Chat input ──────────────────────────────────────────────────
    prompt = st.chat_input("Ask me anything…")

    # ── Get current conversation ──────────────────────────────────────
    conv = _current_conv()
    msgs = conv["messages"]

    # ── Welcome screen or chat history ──────────────────────────────
    if not msgs and not prompt:
        _render_welcome()
    else:
        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    meta = msg["meta"]
                    src = _render_sources(meta.get("faq_sources", []), meta.get("web_sources", []))
                    if src:
                        st.markdown(src, unsafe_allow_html=True)

    # ── Handle new prompt ─────────────────────────────────────────────
    if prompt:
        # Auto-title the conversation from the first message
        if not msgs:
            conv["title"] = prompt[:42].strip()

        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder  = st.empty()
            sources_slot = st.empty()

            with st.spinner("Thinking…"):
                try:
                    data = _call_api(prompt, uid)
                except httpx.HTTPStatusError as e:
                    err = f"**API error {e.response.status_code}:** {e.response.text[:400]}"
                    placeholder.markdown(err)
                    msgs.append({"role": "assistant", "content": err})
                    st.stop()
                except httpx.RequestError as e:
                    err = (
                        "**Cannot reach the API.** Is the backend running?\n\n"
                        f"`{e}`\n\n"
                        "Start it with:\n```\nuvicorn app.main:app --reload --port 8000\n```"
                    )
                    placeholder.markdown(err)
                    msgs.append({"role": "assistant", "content": err})
                    st.stop()
                except Exception as e:  # noqa: BLE001
                    err = f"**Unexpected error:** `{e}`"
                    placeholder.markdown(err)
                    msgs.append({"role": "assistant", "content": err})
                    st.stop()

            answer      = data.get("response") or "_No response returned._"
            model_used  = data.get("model_used", "unknown")
            latency_ms  = data.get("latency_ms")
            sources     = data.get("sources", {})
            faq_sources = sources.get("faq", [])
            web_sources = sources.get("web", [])

            _stream_text(placeholder, answer)
            src = _render_sources(faq_sources, web_sources)
            if src:
                sources_slot.markdown(src, unsafe_allow_html=True)

            # Update per-conversation stats
            conv["total_queries"] += 1
            conv["last_latency"]   = latency_ms
            if model_used in conv["model_counts"]:
                conv["model_counts"][model_used] += 1

            msgs.append({
                "role":    "assistant",
                "content": answer,
                "meta": {
                    "model_used":  model_used,
                    "latency_ms":  latency_ms,
                    "faq_sources": faq_sources,
                    "web_sources": web_sources,
                },
            })
            
            # Save history to disk after every interaction
            _save_history(st.session_state.conversations)


if __name__ == "__main__":
    main()
