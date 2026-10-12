"""What /ask and /health do, shared by the FastAPI routes and the hosted Streamlit app.

Hosted on Streamlit Community Cloud there is no separate API server: the chat page
calls these functions directly (GOLF_BACKEND=inprocess). Locally the API calls them.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import threading

from app.config import settings
from app.models.schemas import AskResponse
from app.rag.pipeline import AssistantPipeline


class DailyLimitReachedError(RuntimeError):
    """The demo's questions-per-day cap is used up."""


class DailyLimiter:
    """Caps questions per UTC day across every visitor of this process (0 = no cap).

    Counts live in memory, so a restart resets them. That is fine for a demo whose
    point is to keep a public link from running up the API bill.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0

    def try_acquire(self) -> bool:
        if self.limit <= 0:
            return True
        with self._lock:
            self._roll_over()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int | None:
        if self.limit <= 0:
            return None
        with self._lock:
            self._roll_over()
            return self.limit - self._used

    def _roll_over(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day, self._used = today, 0


daily_limiter = DailyLimiter(settings.daily_question_limit)
_pipeline: AssistantPipeline | None = None


def get_pipeline() -> AssistantPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AssistantPipeline()
    return _pipeline


def answer(question: str, history: list[dict[str, str]]) -> AskResponse:
    if not daily_limiter.try_acquire():
        raise DailyLimitReachedError(
            f'This demo answers {daily_limiter.limit} questions a day and has reached that '
            'limit. It resets at midnight UTC.'
        )
    return get_pipeline().ask(question, history=history)


def missing_keys() -> list[str]:
    """API keys the configured providers need but don't have, by environment name."""
    gemini = settings.llm_provider.strip().lower() == 'gemini'
    needed = {
        'JINA_API_KEY': settings.jina_api_key,
        'PINECONE_API_KEY': settings.pinecone_api_key,
        'GOOGLE_API_KEY' if gemini else 'ANTHROPIC_API_KEY': (
            settings.google_api_key if gemini else settings.anthropic_api_key
        ),
    }
    return [name for name, value in needed.items() if not value]


def status() -> dict:
    gemini = settings.llm_provider.strip().lower() == 'gemini'
    return {
        'status': 'ok',
        'phase': '2-routed',
        'llm_provider': settings.llm_provider,
        'llm_model': settings.gemini_generation_model if gemini else settings.anthropic_generation_model,
        'router': 'on' if settings.router_enabled else 'off',
        'live_search': 'on' if settings.live_search_enabled and gemini else 'off',
        'missing_keys': missing_keys(),
        'daily_limit': daily_limiter.limit,
        'questions_left_today': daily_limiter.remaining(),
    }
