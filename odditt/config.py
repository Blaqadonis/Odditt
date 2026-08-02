"""Central configuration for the Odditt RAG pipeline.

Extracted verbatim from the notebook's Section 3 (Configuration) cell -- values unchanged.
"""

CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "microsoft/Phi-4-mini-instruct",
    "retriever_k": 8,
    "chunk_size": 350,
    "chunk_overlap": 75,
    "max_new_tokens": 512,
    "do_sample": False,          # deterministic answers; set True + add "temperature" to sample
    "app_title": "🔎 Odditt — Auditable Document Intelligence",
    "guardrail_message": "I'm sorry, but I can only answer questions about the uploaded document(s).",
    "unknown_message": "I don't know based on the information in the uploaded document(s).",
}
