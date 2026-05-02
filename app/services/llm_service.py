"""
llm_service.py - Multi-model LLM service for SupportPlus AI.

Fallback chain:
  1. Gemini (PRIMARY)   — google-genai SDK
  2. Groq   (FALLBACK)  — groq SDK (llama3-8b-8192)
  3. FAQ + Web Response — structured fallback (never crashes)

Flow inside safe_generate():
  Step 1 → Try Gemini
  Step 2 → If Gemini fails (rate limit / any error): Try Groq
  Step 3 → If Groq fails: Return structured FAQ + Firecrawl response

SDK Docs:
  Gemini: google-genai >= 1.0.0  (async via client.aio)
  Groq:   groq >= 0.4.0          (async via AsyncGroq)

Python: 3.10+
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

# ── Gemini ────────────────────────────────────────────────────────────────────
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import AsyncGroq, APIError as GroqAPIError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    AsyncGroq = None          # type: ignore[assignment,misc]
    GroqAPIError = Exception  # type: ignore[assignment,misc]

# ── Config ────────────────────────────────────────────────────────────────────
from app.config import (
    # Gemini
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_TOKENS,
    GEMINI_TEMPERATURE,
    # Groq
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    # Shared LLM tuning
    LLM_FAQ_TOP_K,
    LLM_WEB_TOP_K,
    LLM_MEMORY_TOP_K,
    LLM_MAX_CHARS_PER_FAQ,
    LLM_MAX_CHARS_PER_WEB,
    LLM_MAX_CHARS_PER_MEM,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_SLEEP,
    FAQ_FALLBACK_MIN_SCORE,
)

logger = logging.getLogger("supportplus.llm")

# ---------------------------------------------------------------------------
# Gemini — preferred model fallback chain (newest → oldest)
# ---------------------------------------------------------------------------
_PREFERRED_MODELS: List[str] = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

# ---------------------------------------------------------------------------
# Shared system prompt (used by both Gemini and Groq)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are SupportPlus AI — a friendly, expert customer support assistant.
Your job is to give clear, natural, helpful answers — like a knowledgeable human support agent, not a robot.

## TONE & STYLE RULES (follow strictly)
1. **Sound human** — write conversationally. Use phrases like "Here's how:", "You can try:", "The quickest way is:"
2. **Never say** "According to FAQ 1", "FAQ 2 states", "Based on chunk", or any internal reference numbers.
   → Just present the information naturally as your own knowledge.
3. **Be concise** — use short paragraphs, bullet points, and numbered steps. Avoid walls of text.
4. **Use headings** for multi-part answers (e.g. ## Reset Your Password).
5. **Bold key terms** — highlight important words, actions, or warnings.
6. **Be honest** — if you don't know, say so clearly and suggest contacting support.
7. **Never fabricate** — only use information from the provided context.
8. **Always end with:** "Let me know if there's anything else I can help with! 😊"

## RESPONSE FORMAT (mandatory)
Structure every response exactly like this:

[Your natural, conversational answer here — use headings, bullets, numbered steps]

> If no FAQ or web sources are available for a section, omit that section entirely.
> Do NOT mix source content inside the main answer.
"""

# ---------------------------------------------------------------------------
# Static response strings
# ---------------------------------------------------------------------------
_GEMINI_EMPTY_RESPONSE = (
    "The assistant returned no text for this request (it may have been filtered). "
    "Please try rephrasing your question or try again in a moment."
)


# ---------------------------------------------------------------------------
# Utility — trim text to max_chars
# ---------------------------------------------------------------------------

def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Fallback response builder (FAQ + Web — used when BOTH LLMs fail)
# ---------------------------------------------------------------------------

def _build_fallback_response(
    _query: str,
    faq_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Structured fallback response shown when both Gemini AND Groq are unavailable.

    Format:
      ⚠️ AI service is temporarily unavailable.

      Here's what we found:

      **From FAQ:**
      • [question + answer]

      **From Web:**
      • [title + snippet + URL]

      **Suggestion:**
      • Try again later or follow above steps
    """
    m = metrics if metrics is not None else {}

    # ── FAQ section ──────────────────────────────────────────────────────────
    faq_lines: List[str] = []
    for item in (faq_results or [])[:2]:           # show max 2 FAQ hits
        score = float(item.get("score") or 0.0)
        if score < FAQ_FALLBACK_MIN_SCORE:
            continue
        q = _trim_text(item.get("question") or "", 200)
        a = _trim_text(item.get("answer") or item.get("text") or "", 400)
        section = item.get("section", "")
        sec_tag = f" _(Section: {section})_" if section else ""
        faq_lines.append(f"• **{q}**{sec_tag}\n  {a}")

    # ── Web section ──────────────────────────────────────────────────────────
    web_lines: List[str] = []
    for item in (web_results or [])[:1]:            # show max 1 web result
        title   = _trim_text(item.get("title", "Untitled"), 120)
        snippet = _trim_text(item.get("snippet", ""), 300)
        url     = item.get("url", "")
        web_lines.append(f"• **{title}**\n  {snippet}\n  🔗 {url}")

    # ── Determine fallback_kind for metrics ──────────────────────────────────
    if faq_lines or web_lines:
        m["fallback_kind"] = "faq_web_structured"
    else:
        m["fallback_kind"] = "generic_no_data"

    # ── Assemble response ─────────────────────────────────────────────────────
    parts: List[str] = [
        "⚠️ **AI service is temporarily unavailable.**\n",
        "Here's what we found:\n",
    ]

    if faq_lines:
        parts.append("**From FAQ:**\n" + "\n\n".join(faq_lines))
    else:
        parts.append("**From FAQ:**\n• _No closely matching FAQ entry found._")

    if web_lines:
        parts.append("\n**From Web:**\n" + "\n\n".join(web_lines))
    else:
        parts.append("\n**From Web:**\n• _No web results available._")

    parts.append(
        "\n**Suggestion:**\n"
        "• Try again in a few minutes — the full AI assistant usually gives the best answer.\n"
        "• Rephrase with more detail (product name, error text, what you were doing).\n"
        "• Email **support@supportplus.ai** with screenshots or a short description.\n\n"
        "Let me know if you need more help! 😊"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper — list available Gemini models
# ---------------------------------------------------------------------------

def list_available_models() -> List[str]:
    """
    Return all Gemini model names available on your API key
    that support generateContent. Used by _resolve_model().
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        logger.debug("Available Gemini models: %s", models)
        return models
    except Exception as exc:
        logger.warning("Could not list Gemini models: %s", exc)
        return []


def _resolve_model(requested: str) -> str:
    """
    Pick the best available Gemini model.

    Priority:
      1. Model set in GEMINI_MODEL env var (if available on the API)
      2. Models in _PREFERRED_MODELS order
      3. First model returned by the API
      4. Hard fallback to requested model string
    """
    available = list_available_models()

    if not available:
        logger.warning(
            "Could not fetch model list from Gemini — using '%s' directly.", requested
        )
        return requested

    clean = requested.replace("models/", "")
    if clean in available:
        logger.info("Using configured Gemini model: %s", clean)
        return clean

    for candidate in _PREFERRED_MODELS:
        if candidate in available:
            logger.info(
                "Configured model '%s' unavailable → falling back to '%s'.",
                requested, candidate,
            )
            return candidate

    first = available[0]
    logger.warning("No preferred model found → using first available: %s", first)
    return first


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    """
    Multi-model LLM service: Gemini (primary) → Groq (fallback) → FAQ+Web.

    - Singleton pattern (see get_llm_service() below)
    - Async generation — fully FastAPI compatible
    - Auto Gemini model resolution / fallback chain
    - Groq used transparently when Gemini fails
    - Never raises from safe_generate() — always returns user-visible text
    """

    def __init__(self) -> None:
        # ── Gemini client ─────────────────────────────────────────────────────
        if GEMINI_API_KEY:
            self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            self._model_name: str = _resolve_model(GEMINI_MODEL)
            self._gen_config = types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=GEMINI_TEMPERATURE,
                candidate_count=1,
            )
            logger.info(
                "Gemini ready | model=%s | max_tokens=%d | temperature=%s",
                self._model_name, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE,
            )
        else:
            self._gemini_client = None  # type: ignore[assignment]
            self._model_name = "none"
            self._gen_config = None     # type: ignore[assignment]
            logger.warning("Gemini disabled — GEMINI_API_KEY not set.")

        # ── Groq client ───────────────────────────────────────────────────────
        if _GROQ_AVAILABLE and GROQ_API_KEY:
            self._groq_client: Optional[AsyncGroq] = AsyncGroq(api_key=GROQ_API_KEY)
            logger.info(
                "Groq ready | model=%s | max_tokens=%d | temperature=%s",
                GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
            )
        else:
            self._groq_client = None
            if not _GROQ_AVAILABLE:
                logger.warning("Groq SDK not installed — run: pip install groq")
            else:
                logger.warning("Groq disabled — GROQ_API_KEY not set.")

    # ------------------------------------------------------------------ #
    #  Context clipping (token / cost control)                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def clip_context_for_llm(
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        memory_context: str,
        lightweight: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Shrink what we send to the LLM:
          • FAQ:    top 1–2 results (LLM_FAQ_TOP_K)
          • Web:    at most 1 result (LLM_WEB_TOP_K) — 0 in lightweight mode
          • Memory: trimmed to budget chars
        """
        faq_cap = min(max(LLM_FAQ_TOP_K, 1), 2)
        web_cap = 0 if lightweight else min(max(LLM_WEB_TOP_K, 0), 1)

        faq_out: List[Dict[str, Any]] = []
        for item in (faq_results or [])[:faq_cap]:
            row = dict(item)
            row["text"] = _trim_text(str(item.get("text", "")), LLM_MAX_CHARS_PER_FAQ)
            if item.get("answer"):
                row["answer"] = _trim_text(str(item["answer"]), LLM_MAX_CHARS_PER_FAQ * 2)
            if item.get("question"):
                row["question"] = _trim_text(str(item["question"]), 200)
            faq_out.append(row)

        web_out: List[Dict[str, Any]] = []
        for item in (web_results or [])[:web_cap]:
            row = dict(item)
            row["snippet"] = _trim_text(str(item.get("snippet", "")), LLM_MAX_CHARS_PER_WEB)
            row["title"]   = _trim_text(str(item.get("title", "")), 120)
            web_out.append(row)

        mem_budget = max(200, LLM_MAX_CHARS_PER_MEM * 6)
        mem_out = _trim_text(memory_context or "", mem_budget)

        return faq_out, web_out, mem_out

    # ------------------------------------------------------------------ #
    #  Prompt builder (shared by Gemini AND Groq)                         #
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        query: str,
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """
        Build a rich, grounded prompt from all context sources.
        Returns a single string used by both Gemini and Groq.
        """
        sections: List[str] = []

        # ── Memory ───────────────────────────────────────────────────────
        if memory_context.strip():
            sections.append(
                "## 🧠 Previous Interactions (User Memory)\n"
                + memory_context.strip()
            )

        # ── FAQ / RAG ─────────────────────────────────────────────────────
        if faq_results:
            lines = ["## 📚 FAQ Knowledge Base"]
            for i, item in enumerate(faq_results, 1):
                sec   = f" | Section: {item['section']}" if item.get("section") else ""
                score = item.get("score", 0.0)
                q     = (item.get("question") or "").strip()
                body  = (item.get("answer") or item.get("text") or "").strip()
                block = f"**Q:** {q}\n**A:** {body}" if q else body
                lines.append(f"\n### FAQ {i}{sec} (relevance: {score:.2f})\n{block}")
            sections.append("\n".join(lines))
        else:
            sections.append(
                "## 📚 FAQ Knowledge Base\n"
                "_No relevant FAQ entries found for this query._"
            )

        # ── Web results (Firecrawl) ───────────────────────────────────────
        if web_results:
            lines = ["## 🌐 Live Web Information (Firecrawl)"]
            for i, item in enumerate(web_results, 1):
                lines.append(
                    f"\n### Web {i}: {item.get('title', 'Untitled')}\n"
                    f"**Source:** {item.get('url', '')}\n"
                    f"{item.get('snippet', '').strip()}"
                )
            sections.append("\n".join(lines))
        else:
            sections.append(
                "## 🌐 Live Web Information\n"
                "_No web results available._"
            )

        # ── User question ─────────────────────────────────────────────────
        q = _trim_text(query.strip(), 2000)
        sections.append(
            "---\n"
            "## User Question\n"
            f"{q}\n\n"
            "**Your task:**\n"
            "1. Answer the question naturally and conversationally — no robotic phrasing.\n"
            "2. Do NOT mention FAQ numbers, chunk IDs, or index numbers anywhere in your answer.\n"
            "3. Use the FAQ and web data as your knowledge — present it as if you already know it.\n"
            "4. Structure: use headings, bullets, or numbered steps where helpful.\n"
            "5. End with your standard friendly closing."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------ #
    #  Step 1: Generate with Gemini (PRIMARY)                             #
    # ------------------------------------------------------------------ #

    async def generate_with_gemini(
        self,
        query: str,
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        memory_context: str = "",
        _retries: Optional[int] = None,
    ) -> str:
        """
        Generate a response using Google Gemini.

        Raises ValueError on all unrecoverable errors (safe_generate catches this
        and falls through to Groq).
        """
        if not self._gemini_client:
            raise ValueError("Gemini is not configured (GEMINI_API_KEY missing).")

        if _retries is None:
            _retries = LLM_MAX_RETRIES

        prompt = self.build_prompt(query, faq_results, web_results, memory_context)

        logger.debug(
            "Calling Gemini [%s] | faq=%d | web=%d | memory=%s | prompt=%d chars",
            self._model_name,
            len(faq_results),
            len(web_results),
            "yes" if memory_context.strip() else "no",
            len(prompt),
        )

        try:
            response = await self._gemini_client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._gen_config,
            )
            answer = self._safe_text(response)
            logger.info(
                "✅ Gemini OK | model=%s | response=%d chars",
                self._model_name, len(answer),
            )
            return answer

        except ClientError as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            msg    = str(exc)
            logger.error("Gemini ClientError [%s]: %s", status, msg)

            # Rate limit — retry once with backoff, then give up to Groq
            if status == 429 or "quota" in msg.lower() or "rate" in msg.lower():
                retry_after = min(LLM_RETRY_BASE_DELAY * 2.0, LLM_RETRY_MAX_SLEEP)
                m = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
                if m:
                    retry_after = min(float(int(m.group(1)) + 1), LLM_RETRY_MAX_SLEEP)

                if _retries > 0:
                    logger.warning(
                        "Gemini rate-limited — waiting %ds then retrying (%d retries left)…",
                        retry_after, _retries,
                    )
                    await asyncio.sleep(retry_after)
                    return await self.generate_with_gemini(
                        query, faq_results, web_results, memory_context,
                        _retries=_retries - 1,
                    )

                raise ValueError(
                    f"Gemini rate-limited (status={status}). "
                    "Free-tier quota may be exhausted — trying Groq fallback."
                ) from exc

            if status in (404, 400) or "not found" in msg.lower() or "invalid" in msg.lower():
                logger.warning("Gemini model '%s' rejected — attempting re-resolution.", self._model_name)
                self._reinitialise_gemini()
                raise ValueError(
                    f"Gemini model '{self._model_name}' was unavailable. Falling back to Groq."
                ) from exc

            raise ValueError(f"Gemini API error ({status}): {msg}") from exc

        except APIError as exc:
            logger.error("Gemini APIError: %s", exc)
            raise ValueError(f"Gemini API error: {exc}") from exc

        except asyncio.TimeoutError:
            logger.error("Gemini request timed out.")
            raise ValueError("Gemini timed out. Trying Groq fallback.") from None

        except Exception as exc:
            logger.error("Unexpected Gemini error: %s", exc, exc_info=True)
            raise ValueError(f"Unexpected Gemini error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Step 2: Generate with Groq (FALLBACK)                              #
    # ------------------------------------------------------------------ #

    async def generate_with_groq(
        self,
        query: str,
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """
        Generate a response using Groq (llama3-8b-8192).

        Reuses the same prompt structure as Gemini.
        Groq uses the OpenAI-compatible chat format:
          - system message  = _SYSTEM_PROMPT
          - user message    = build_prompt() output (context + question)

        Raises ValueError on all errors (safe_generate catches and falls to FAQ+Web).
        """
        if not self._groq_client:
            raise ValueError("Groq is not configured (GROQ_API_KEY missing or SDK not installed).")

        prompt = self.build_prompt(query, faq_results, web_results, memory_context)

        logger.debug(
            "Calling Groq [%s] | faq=%d | web=%d | memory=%s | prompt=%d chars",
            GROQ_MODEL,
            len(faq_results),
            len(web_results),
            "yes" if memory_context.strip() else "no",
            len(prompt),
        )

        try:
            chat_completion = await self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=GROQ_MAX_TOKENS,
                temperature=GROQ_TEMPERATURE,
            )

            answer = chat_completion.choices[0].message.content or ""
            answer = answer.strip()

            if not answer:
                logger.warning("Groq returned an empty response.")
                raise ValueError("Groq returned an empty response.")

            logger.info("✅ Groq OK | model=%s | response=%d chars", GROQ_MODEL, len(answer))
            return answer

        except GroqAPIError as exc:
            status = getattr(exc, "status_code", None)
            logger.error("Groq APIError [%s]: %s", status, exc)
            raise ValueError(f"Groq API error ({status}): {exc}") from exc

        except asyncio.TimeoutError:
            logger.error("Groq request timed out.")
            raise ValueError("Groq timed out.") from None

        except Exception as exc:
            logger.error("Unexpected Groq error: %s", exc, exc_info=True)
            raise ValueError(f"Unexpected Groq error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  safe_generate — 3-tier fallback, NEVER raises                      #
    # ------------------------------------------------------------------ #

    async def safe_generate(
        self,
        query: str,
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        memory_context: str = "",
        lightweight: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Always returns user-visible text — never raises an exception.

        Fallback chain:
          1. Gemini  → success → return (metrics["llm"] = "gemini")
          2. Groq    → success → return (metrics["llm"] = "groq")
          3. FAQ+Web → structured response (metrics["llm"] = "fallback")

        Args:
            query:          User's question.
            faq_results:    RAG hits from ChromaDB.
            web_results:    Firecrawl results (always passed, even in fallback).
            memory_context: Formatted past interactions string.
            lightweight:    If True, skip web results in the prompt.
            metrics:        Dict updated in-place with timing and model info.

        Returns:
            Response string to show the user.
        """
        m = metrics if metrics is not None else {}
        m["lightweight"] = lightweight

        q = _trim_text(query.strip(), 2000)

        # Clip context (shared preparation for both LLMs)
        faq_c, web_c, mem_c = self.clip_context_for_llm(
            faq_results, web_results, memory_context, lightweight
        )
        m["context_sizes"] = {
            "faq_chunks":    len(faq_c),
            "web_chunks":    len(web_c),
            "memory_chars":  len(mem_c),
        }

        # ── Step 1: Try Groq (PRIMARY) ────────────────────────────────────
        t0 = time.perf_counter()
        try:
            text = await self.generate_with_groq(q, faq_c, web_c, mem_c)
            m["llm"]        = "groq"
            m["model_used"] = GROQ_MODEL
            m["groq_ms"]    = round((time.perf_counter() - t0) * 1000, 2)
            logger.info("LLM used: Groq | %.2f ms", m["groq_ms"])
            return text

        except Exception as groq_exc:
            m["groq_error"] = str(groq_exc)
            m["groq_ms"]    = round((time.perf_counter() - t0) * 1000, 2)
            logger.warning(
                "⚠️  Groq failed (%.0f ms) — trying Gemini fallback. Error: %s",
                m["groq_ms"], groq_exc,
            )

        # ── Step 2: Try Gemini (FALLBACK) ─────────────────────────────────
        t1 = time.perf_counter()
        try:
            text = await self.generate_with_gemini(q, faq_c, web_c, mem_c)
            m["llm"]            = "gemini"
            m["model_used"]     = self._model_name
            m["gemini_ms"]      = round((time.perf_counter() - t1) * 1000, 2)
            logger.info("LLM used: Gemini | %.2f ms", m["gemini_ms"])
            return text

        except Exception as gemini_exc:
            m["gemini_error"] = str(gemini_exc)
            m["gemini_ms"]    = round((time.perf_counter() - t1) * 1000, 2)
            logger.warning(
                "⚠️  Gemini failed (%.0f ms) — using FAQ+Web fallback. Error: %s",
                m["gemini_ms"], gemini_exc,
            )

        # ── Step 3: FAQ + Web structured fallback ─────────────────────────
        logger.error(
            "❌ Both Gemini and Groq failed. Serving structured FAQ+Web fallback. "
            "gemini_err=%r  groq_err=%r",
            m.get("gemini_error", "n/a"),
            m.get("groq_error", "n/a"),
        )
        m["llm"]        = "fallback"
        m["model_used"] = "none"

        # Pass ORIGINAL (unclipped) results to the fallback so the user
        # sees the most complete FAQ + web information available.
        faq_display, _, _ = self.clip_context_for_llm(
            faq_results, [], "", lightweight=False
        )
        return _build_fallback_response(q, faq_display, web_results, metrics=m)

    # ------------------------------------------------------------------ #
    #  Source ranking                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def rank_sources(
        faq_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        faq_weight: float = 0.7,
        web_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """Compute weighted confidence score for logging + response metadata."""
        faq_top = max((r.get("score", 0.0) for r in faq_results), default=0.0)
        web_top = max((r.get("score", 0.0) for r in web_results), default=0.0)
        return {
            "faq_top_score":       round(faq_top, 4),
            "web_top_score":       round(web_top, 4),
            "combined_confidence": round(faq_weight * faq_top + web_weight * web_top, 4),
            "faq_chunks":          len(faq_results),
            "web_results":         len(web_results),
        }

    # ------------------------------------------------------------------ #
    #  Health check                                                        #
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        """Return health status of both Gemini and Groq connections."""
        return {
            "status":           "ok",
            # Gemini
            "primary_model":    self._model_name,
            "gemini_available": bool(self._gemini_client),
            "gemini_key_set":   bool(GEMINI_API_KEY),
            # Groq
            "fallback_model":   GROQ_MODEL,
            "groq_available":   bool(self._groq_client),
            "groq_key_set":     bool(GROQ_API_KEY),
            "groq_sdk_installed": _GROQ_AVAILABLE,
            # Shared
            "max_tokens_gemini": GEMINI_MAX_TOKENS,
            "max_tokens_groq":   GROQ_MAX_TOKENS,
            "temperature":       GEMINI_TEMPERATURE,
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_text(response: Any) -> str:
        """Extract text safely from a Gemini GenerateContentResponse."""
        try:
            txt = response.text
            if txt and txt.strip():
                return txt.strip()
        except (AttributeError, ValueError):
            pass

        try:
            for candidate in (response.candidates or []):
                parts = candidate.content.parts or []
                combined = " ".join(
                    p.text for p in parts if getattr(p, "text", None)
                ).strip()
                if combined:
                    return combined
        except Exception:
            pass

        logger.warning("Gemini returned an empty or blocked response.")
        return _GEMINI_EMPTY_RESPONSE

    def _reinitialise_gemini(self) -> None:
        """Re-resolve + rebuild the Gemini model after a 404/400 error."""
        try:
            self._model_name = _resolve_model(GEMINI_MODEL)
            logger.info("Gemini re-initialised → model: %s", self._model_name)
        except Exception as exc:
            logger.error("Gemini re-initialisation failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return the shared LLMService singleton (lazy-initialised)."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ---------------------------------------------------------------------------
# Standalone demo / debug
# Run:  python -m app.services.llm_service
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import sys

    # Force UTF-8 output on Windows so emoji in responses prints correctly
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    _DEMO_FAQ = [
        {
            "text": (
                "To reset your password, go to the login page and click "
                "'Forgot Password'. Enter your registered email and follow "
                "the reset link sent to your inbox."
            ),
            "question": "How do I reset my password?",
            "section":  "Account & Billing",
            "score":    0.94,
            "answer": (
                "Go to the login page → click 'Forgot Password' → "
                "enter your email → follow the reset link in your inbox."
            ),
        }
    ]
    _DEMO_WEB = [
        {
            "title":   "SupportPlus Password Help",
            "url":     "https://docs.supportplus.ai/password-reset",
            "snippet": "Step-by-step guide to reset your SupportPlus account password securely.",
            "score":   0.80,
        }
    ]
    _DEMO_MEM = (
        "## Previous Interactions\n"
        "User reported login difficulties yesterday and was advised to clear cache."
    )

    async def _demo() -> None:
        svc = get_llm_service()
        print(f"\n=== Health ===")
        import json
        print(json.dumps(svc.health(), indent=2))

        metrics: dict = {}
        print("\n=== Running safe_generate (Gemini -> Groq -> Fallback) ===\n")
        answer = await svc.safe_generate(
            query="How do I reset my password?",
            faq_results=_DEMO_FAQ,
            web_results=_DEMO_WEB,
            memory_context=_DEMO_MEM,
            metrics=metrics,
        )
        print("=== Response ===")
        print(answer)
        print("\n=== Pipeline Metrics ===")
        print(json.dumps(metrics, indent=2))

    asyncio.run(_demo())
