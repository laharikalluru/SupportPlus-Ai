"""
web_service.py - Real-time web search integration via Firecrawl API.

Falls back gracefully to a mock response when:
  - WEB_SEARCH_ENABLED is False
  - FIRECRAWL_API_KEY is not configured
  - The API call fails (network error, rate limit, etc.)
"""

import logging
import httpx
from typing import List, Dict, Any, Optional

from app.config import (
    FIRECRAWL_API_KEY,
    FIRECRAWL_BASE_URL,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_TIMEOUT,
)

logger = logging.getLogger("supportplus.web")


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

def _make_web_result(
    title: str,
    url: str,
    snippet: str,
    source: str = "web",
    score: float = 0.0,
) -> Dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
        "score": round(score, 4),
    }


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

_MOCK_RESULTS: List[Dict[str, Any]] = [
    _make_web_result(
        title="SupportPlus AI – Official Documentation",
        url="https://docs.supportplus.ai",
        snippet=(
            "Comprehensive guides, API references, and tutorials for SupportPlus AI. "
            "Learn how to set up integrations, manage your knowledge base, and optimise "
            "AI response quality."
        ),
        source="mock",
        score=0.85,
    ),
    _make_web_result(
        title="SupportPlus AI Community Forum – Latest Discussions",
        url="https://community.supportplus.ai",
        snippet=(
            "Join thousands of SupportPlus users discussing tips, troubleshooting steps, "
            "best practices, and feature requests. Search for your question before posting."
        ),
        source="mock",
        score=0.78,
    ),
    _make_web_result(
        title="SupportPlus AI Status Page",
        url="https://status.supportplus.ai",
        snippet=(
            "Real-time system status for all SupportPlus AI services including API, "
            "chat widget, knowledge base indexing, and third-party integrations."
        ),
        source="mock",
        score=0.70,
    ),
]


# ---------------------------------------------------------------------------
# Firecrawl client
# ---------------------------------------------------------------------------

class WebService:
    """
    Encapsulates web search via the Firecrawl API.

    Firecrawl /search endpoint accepts:
      POST /search  { "query": str, "limit": int }

    Response:
      { "data": [ { "title": str, "url": str, "markdown": str, ... } ] }
    """

    def __init__(self) -> None:
        self._enabled = WEB_SEARCH_ENABLED and bool(FIRECRAWL_API_KEY)
        if not WEB_SEARCH_ENABLED:
            logger.info("Web search is disabled (WEB_SEARCH_ENABLED=false).")
        elif not FIRECRAWL_API_KEY:
            logger.warning(
                "Web search enabled but FIRECRAWL_API_KEY not set – mock data will be used."
            )
        else:
            logger.info("WebService initialised (Firecrawl API ready).")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Perform a web search for `query` and return up to `limit` results.

        Each result dict contains: title, url, snippet, source, score.
        """
        if not self._enabled:
            logger.debug("Web search skipped – returning mock results.")
            return _MOCK_RESULTS[:limit]

        try:
            return await self._firecrawl_search(query, limit)
        except httpx.TimeoutException:
            logger.warning("Firecrawl request timed out – using mock results.")
            return _MOCK_RESULTS[:limit]
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Firecrawl HTTP error %d – using mock results. Detail: %s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            return _MOCK_RESULTS[:limit]
        except Exception as exc:
            logger.error("Web search failed: %s", exc, exc_info=True)
            return _MOCK_RESULTS[:limit]

    # ------------------------------------------------------------------ #
    #  Firecrawl implementation                                           #
    # ------------------------------------------------------------------ #

    async def _firecrawl_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Call the Firecrawl /search endpoint and normalise the response."""
        url = f"{FIRECRAWL_BASE_URL}/search"
        headers = {
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "limit": limit}

        async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_results = data.get("data", [])
        if not raw_results:
            logger.info("Firecrawl returned no results for query: '%s'", query[:80])
            return []

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_results[:limit]):
            # `markdown` field contains full page text; truncate to snippet
            full_text: str = item.get("markdown", item.get("content", ""))
            snippet = self._extract_snippet(full_text, max_chars=400)

            results.append(_make_web_result(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=snippet,
                source="firecrawl",
                # Firecrawl doesn't return scores; use rank-based proxy
                score=round(1.0 - (idx * 0.1), 2),
            ))

        logger.info("Firecrawl returned %d results for query: '%s'", len(results), query[:80])
        return results

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_snippet(text: str, max_chars: int = 400) -> str:
        """Return the first `max_chars` characters of `text`, stripped cleanly."""
        if not text:
            return ""
        # Remove markdown headings and excessive whitespace
        import re
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"\n{2,}", " ", text).strip()
        if len(text) <= max_chars:
            return text
        # Try to end at a sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars // 2:
            return truncated[: last_period + 1]
        return truncated + "…"

    def format_web_context(self, results: List[Dict[str, Any]]) -> str:
        """Format web results into a readable context block for the LLM prompt."""
        if not results:
            return ""
        lines = ["## Live Web Information"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] **{r['title']}** ({r['url']})")
            lines.append(f"    {r['snippet']}")
        return "\n".join(lines)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "web_search_enabled": WEB_SEARCH_ENABLED,
            "firecrawl_configured": bool(FIRECRAWL_API_KEY),
            "using_mock": not self._enabled,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_web_service: Optional[WebService] = None


def get_web_service() -> WebService:
    global _web_service
    if _web_service is None:
        _web_service = WebService()
    return _web_service
