from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sys
import time

import httpx
import streamlit as st


def _export_secrets() -> None:
    """Streamlit Cloud keeps settings in st.secrets; app.config reads environment variables."""
    try:
        secrets = dict(st.secrets)
    except Exception:                                      # no secrets file: local run
        return
    for name, value in secrets.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(name, str(value))


_export_secrets()
# api: call the FastAPI server at GOLF_API_URL (local runs, start.bat).
# inprocess: run the pipeline inside this app, for hosts with no API server (Streamlit Cloud).
BACKEND = os.environ.get('GOLF_BACKEND', 'api').strip().lower()
if BACKEND == 'inprocess':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
API_URL = os.environ.get('GOLF_API_URL', 'http://127.0.0.1:8001')
ASK_TIMEOUT_SECONDS = 150.0                                # grounded answers can take ~50s
HEALTH_TTL_SECONDS = 30                                    # keeps reruns off the health endpoint
HISTORY_TURNS = 6                                          # the API only uses the last three exchanges

# Knowledge-base questions from the retrieval eval, plus one live and one prediction.
EXAMPLE_QUESTIONS = [
    'What is the difference between stroke play and match play?',
    "What are the four men's major championships?",
    'What is a links golf course?',
    'What is the dimple pattern on a golf ball for?',
    'Who jointly writes the Rules of Golf?',
    'What is a mulligan?',
    "Who is the men's world number one right now?",
    'Who are the favourites for the next Masters?',
]
# Shown above each answer so it is clear where it came from.
ROUTE_LABELS = {
    'static': '📚 From the golf knowledge base',
    'live': '🌐 From a live web search',
    'prediction': "🔮 Published predictions from the web: other people's opinions, not facts or betting advice",
    'off_topic': '⛳ Outside golf',
}
LINK_WITHOUT_TARGET_RE = re.compile(r'<a (?![^>]*\btarget=)')

st.set_page_config(
    page_title='CaddieAI',
    page_icon='⛳',
    layout='centered',
    initial_sidebar_state='expanded',
)


class BackendError(Exception):
    """A message safe and useful to show the user in place of an answer."""


@st.cache_data(ttl=HEALTH_TTL_SECONDS, show_spinner=False)
def fetch_health() -> dict | None:
    """None means the API is unreachable, which disables the composer."""
    if BACKEND == 'inprocess':
        from app import service
        return service.status()
    try:
        response = httpx.get(f'{API_URL}/health', timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def ask_backend(question: str, history: list[dict]) -> dict:
    if BACKEND == 'inprocess':
        return _ask_inprocess(question, history)
    try:
        response = httpx.post(
            f'{API_URL}/ask',
            json={'question': question, 'history': history},
            timeout=ASK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError as exc:
        raise BackendError(f'Cannot reach the API at {API_URL}.') from exc
    except httpx.TimeoutException as exc:
        raise BackendError(f'No response within {ASK_TIMEOUT_SECONDS:.0f}s. Try again in a moment.') from exc
    except httpx.HTTPStatusError as exc:
        raise BackendError(friendly_error(exc)) from exc


def _ask_inprocess(question: str, history: list[dict]) -> dict:
    """The same checks and error messages as the API's /ask, without a server."""
    from pydantic import ValidationError

    from app import service
    from app.models.schemas import AskRequest
    from app.rag.errors import RateLimitedError, UpstreamUnavailableError

    try:
        request = AskRequest(question=question, history=history)
    except ValidationError as exc:
        raise BackendError('That question is too short or too long.') from exc
    try:
        turns = [turn.model_dump() for turn in request.history]
        return service.answer(request.question.strip(), turns).model_dump()
    except (service.DailyLimitReachedError, RateLimitedError, UpstreamUnavailableError) as exc:
        raise BackendError(str(exc)) from exc
    except Exception as exc:
        # Internal messages can carry raw provider payloads: log them, show a generic one.
        logging.getLogger(__name__).exception('Ask failed')
        raise BackendError('Something went wrong while answering. Try again in a moment.') from exc


def friendly_error(exc: httpx.HTTPStatusError) -> str:
    """Reads the API's status codes so a quota problem is not mistaken for a bug."""
    try:
        detail = str(exc.response.json().get('detail') or '')
    except ValueError:
        detail = ''
    code = exc.response.status_code
    if code == 429:
        return detail or 'The provider rate limit or daily quota is exhausted. Try again later.'
    if code == 503:
        return detail or 'The model provider is temporarily unavailable. Try again in a minute.'
    if code == 422:
        return 'That question is too short or too long.'
    return detail or f'The API returned HTTP {code}.'


def render_sources(sources: list[dict]) -> None:
    """Numbers match the [n] markers in the answer so any claim can be traced back."""
    if not sources:
        return
    label = 'Web sources' if sources[0].get('category') == 'web' else 'Sources'
    with st.expander(f'{label} ({len(sources)})'):
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**[{source['n']}]** [{source['title']}](<{source['url']}>)")
                if source.get('category'):
                    st.badge(source['category'], color='green')


def render_search_suggestions(html: str) -> None:
    """Google's terms require its search suggestions alongside every web-grounded answer.
    Links open in a new tab so following one does not end the chat session."""
    if html:
        st.html(LINK_WITHOUT_TARGET_RE.sub('<a target="_blank" rel="noopener" ', html))


def chat_history(messages: list[dict]) -> list[dict]:
    """Earlier turns sent with a question so the API can resolve follow-ups like "he"."""
    return [
        {'role': m['role'], 'content': m['content']}
        for m in messages
        if m.get('content') and not m.get('error')             # failed turns carry nothing useful
    ][-HISTORY_TURNS:]


def render_assistant(message: dict) -> None:
    """Shared by the live turn and the replayed history so both look identical."""
    if message.get('error'):
        st.error(message['content'])
        return
    if message.get('route') in ROUTE_LABELS:
        st.caption(ROUTE_LABELS[message['route']])
    if message.get('search_query'):
        st.caption(f"🔎 Searched for: {message['search_query']}")
    st.markdown(message['content'])
    render_search_suggestions(message.get('search_suggestions_html') or '')
    # Off-topic replies are fixed text, so naming a model would suggest it wrote them.
    show_model = message.get('model') and message.get('route') != 'off_topic'
    footnote = [
        part
        for part in (
            f"{message.get('provider')} · {message.get('model')}" if show_model else '',
            f"{message['elapsed']:.1f}s" if message.get('elapsed') else '',
        )
        if part
    ]
    if footnote:
        st.caption(' — '.join(footnote))
    render_sources(message.get('sources') or [])


if 'messages' not in st.session_state:
    st.session_state.messages = []

health = fetch_health()
# Missing keys only matter when this app runs the pipeline itself (the API checks its own).
missing = (health or {}).get('missing_keys') or []
ready = bool(health) and not missing

st.title('⛳ CaddieAI')
st.caption(
    'Ask about rules, history, courses, tournaments, players, or equipment, or about '
    'what is happening on tour now. Every answer cites the sources it used.'
)

# Read before the examples so they hide on the run that submits the first question.
# In the main area the input stays pinned to the bottom wherever it is called.
typed = st.chat_input(
    'Ask about golf rules, history, courses, players, or equipment…',
    disabled=not ready,
)

if not st.session_state.messages and not typed and 'pending_question' not in st.session_state:
    st.markdown('##### Start with an example')
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(example, key=f'example-{index}', use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_assistant(message)
        else:
            st.markdown(message['content'])

question = typed or st.session_state.pop('pending_question', None)

if question:
    history = chat_history(st.session_state.messages)
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        message: dict = {'role': 'assistant'}
        try:
            with st.spinner('Finding sources and drafting a cited answer…'):
                started = time.perf_counter()
                data = ask_backend(question, history)
                elapsed = time.perf_counter() - started
            search_query = data.get('search_query') or ''
            message.update(
                search_query=search_query if search_query != question else '',
                route=data.get('route', ''),
                search_suggestions_html=data.get('search_suggestions_html') or '',
                content=data.get('answer') or 'No answer returned.',
                sources=data.get('sources') or [],
                provider=data.get('provider', ''),
                model=data.get('model', ''),
                elapsed=elapsed,
            )
        except BackendError as exc:
            message.update(content=str(exc), error=True)
        render_assistant(message)

    st.session_state.messages.append(message)
    fetch_health.clear()                                   # so "questions left today" is current
    health = fetch_health()

# Drawn last, so controls reflect the conversation after this turn was appended.
# Sidebar placement is by container, not script order, so it still renders on the left.
with st.sidebar:
    st.markdown('### ⛳ CaddieAI')
    st.caption('Phase 2 — routed: knowledge base, live web search, or published predictions.')
    st.divider()

    if missing:
        st.error('Not configured')
        st.caption("Add these to the app's secrets: " + ', '.join(missing))
    elif health:
        st.success('Online' if BACKEND == 'inprocess' else 'API online')
        st.markdown(
            f"**Generator**  \n{health.get('llm_provider', 'unknown')} · "
            f"`{health.get('llm_model', 'unknown')}`"
        )
        if health.get('live_search') == 'off':
            st.caption('Live web search is off: live and prediction questions get a short notice.')
        if health.get('questions_left_today') is not None:
            st.caption(
                f"Demo limit: {health['questions_left_today']} of {health['daily_limit']} "
                'questions left today (resets at midnight UTC).'
            )
    else:
        st.error('API offline')
        st.caption(f'Nothing answering at {API_URL}. Start it with:')
        st.code('uvicorn app.main:app --port 8001', language='bash')
        # Seen on the first deploy: secrets not read, so the hosted app looked for a local API.
        st.caption('Hosted, e.g. on Streamlit Cloud? Add `GOLF_BACKEND = "inprocess"` to the app\'s secrets and reboot it.')
        if st.button('Retry connection', use_container_width=True):
            fetch_health.clear()
            st.rerun()

    st.divider()
    if st.button(
        'Clear conversation',
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown('**Scope**')
    st.caption(
        'Each question is routed first. Golf knowledge is answered only from curated '
        'Wikipedia articles. Live scores, rankings and news are answered from a Google '
        'search. Predictions are reported as published opinions, never made up. '
        'Non-golf questions are declined.'
    )
    st.caption('Knowledge base text from Wikipedia, CC BY-SA 4.0. Web answers grounded with Google Search.')
