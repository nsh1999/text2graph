"""Modules package for text2graph."""

from text2graph.modules.db import PostgreSQLConnection
from text2graph.modules.ollama_client import (
    OllamaClient,
    OllamaMessage,
    OllamaResponse,
    OllamaGenerateResponse,
    OllamaError,
    ChatResult,
    chat,
)

__all__ = [
    "PostgreSQLConnection",
    "OllamaClient",
    "OllamaMessage",
    "OllamaResponse",
    "OllamaGenerateResponse",
    "OllamaError",
    "ChatResult",
    "chat",
]
