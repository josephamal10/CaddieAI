from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from app import service
from app.main import app
from app.models.schemas import AskResponse


class _Pipeline:
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, question: str, history: list[dict] | None = None) -> AskResponse:
        self.calls += 1
        return AskResponse(answer='ok', sources=[], route='static', retrieved_chunk_ids=[])


def test_the_daily_limit_counts_every_question_and_resets_the_next_day(monkeypatch) -> None:
    days = iter([datetime(2026, 10, 12, 23, 0, tzinfo=timezone.utc)] * 4
                + [datetime(2026, 10, 13, 0, 1, tzinfo=timezone.utc)] * 2)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(days)

    monkeypatch.setattr(service, 'datetime', _Clock)
    limiter = service.DailyLimiter(2)

    assert [limiter.try_acquire() for _ in range(3)] == [True, True, False]
    assert limiter.remaining() == 0
    assert limiter.try_acquire() and limiter.remaining() == 1     # a new UTC day


def test_no_limit_by_default() -> None:
    limiter = service.DailyLimiter(0)
    assert all(limiter.try_acquire() for _ in range(1000)) and limiter.remaining() is None


def test_ask_returns_429_once_the_daily_limit_is_used_up(monkeypatch) -> None:
    pipeline = _Pipeline()
    monkeypatch.setattr(service, '_pipeline', pipeline)
    monkeypatch.setattr(service, 'daily_limiter', service.DailyLimiter(1))
    client = TestClient(app)

    assert client.post('/ask', json={'question': 'What is a bogey?'}).status_code == 200
    over = client.post('/ask', json={'question': 'What is a birdie?'})

    assert over.status_code == 429
    assert '1 questions a day' in over.json()['detail']
    assert pipeline.calls == 1                             # the provider was never called


@pytest.mark.parametrize('provider, missing', [
    ('gemini', ['GOOGLE_API_KEY']),
    ('claude', ['ANTHROPIC_API_KEY']),
])
def test_health_names_the_missing_keys_without_their_values(monkeypatch, provider, missing) -> None:
    monkeypatch.setattr(service.settings, 'llm_provider', provider)
    monkeypatch.setattr(service.settings, 'jina_api_key', 'j')
    monkeypatch.setattr(service.settings, 'pinecone_api_key', 'p')
    monkeypatch.setattr(service.settings, 'google_api_key', '')
    monkeypatch.setattr(service.settings, 'anthropic_api_key', '')

    health = TestClient(app).get('/health').json()
    assert health['missing_keys'] == missing
    assert 'j' not in health.values() and 'p' not in health.values()
