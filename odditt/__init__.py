"""Odditt -- core RAG pipeline: config, model loading, and the DocChatbot itself.

    from odditt.config import CONFIG
    from odditt.model_loader import load_embeddings, load_llm
    from odditt.chatbot import DocChatbot

See app.py at the repo root for how these are assembled into the Gradio app, and evals/ for the
evaluation framework that scores DocChatbot's answers against a fixed gold question set.
"""
from .chatbot import DocChatbot
from .config import CONFIG
from .model_loader import free_llm, load_embeddings, load_llm

__all__ = ["CONFIG", "DocChatbot", "load_embeddings", "load_llm", "free_llm"]
