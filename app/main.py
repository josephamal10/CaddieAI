from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.logging_config import configure_logging


configure_logging()

app = FastAPI(
    title='CaddieAI',
    version='0.1.0',
    description='Golf knowledge assistant: routed RAG with live web search.',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router)

