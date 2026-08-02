"""Grounding-score helpers -- extracted from the notebook's Section 7 cell."""
import re

_STOPWORDS = set('''
a an the is are was were be been being of to in on for and or but if with as by at from this that
these those it its into about over under than then so not no do does did can could should would
may might will shall you your yours i we our ours they their theirs he she his her him them
'''.split())


def tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9%$.]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
