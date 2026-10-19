"""Decide whether a question belongs to this assistant before any search or LLM call.

The scope is Royal Enfield specifications from the brochure and dealer
contacts from the directory. Unrelated questions get a fixed reply.
"""

from __future__ import annotations

import re

import numpy as np

SCOPE_THRESHOLD = 0.29
OUT_OF_SCOPE = (
    "I can only answer questions about Royal Enfield motorcycles in this brochure "
    "and about Royal Enfield dealers in the directory. Ask about a model, a "
    "specification, a price, or a dealer."
)

DOMAIN_ANCHORS = [
    "Royal Enfield motorcycle specifications engine power torque mileage seat height fuel tank brakes tyres colours variants price",
    "Royal Enfield dealer phone number address showroom contact in an Indian state or district",
    "Bullet 350 Hunter 350 Classic 350 Meteor 350 Goan Classic Himalayan 450 Himalayan 440 Scram 440 Guerrilla 450 Interceptor 650 Continental GT 650 Classic 650 Bullet 650 Bear 650 Shotgun 650 Super Meteor 650",
    "ex-showroom price in rupees kerb weight gearbox ABS wheelbase ground clearance",
    "which Royal Enfield dealer or studio store is in this city",
]

OTHER_BRANDS = re.compile(
    r"\b(honda|yamaha|bajaj|hero|tvs|ktm|bmw|ducati|triumph|jawa|harley|"
    r"suzuki|kawasaki|ather|vespa|aprilia|benelli|ola)\b",
    re.IGNORECASE,
)
PROMPT_INJECTION = re.compile(
    r"ignore (all |any )?(previous|above|prior) instructions|"
    r"reveal (your|the) (system )?prompt|"
    r"\b(jailbreak|developer mode)\b|"
    r"you are now",
    re.IGNORECASE,
)

_anchor_matrix: np.ndarray | None = None


def _anchors(embedder) -> np.ndarray:
    global _anchor_matrix
    if _anchor_matrix is None:
        vectors = embedder.embed_documents(DOMAIN_ANCHORS)
        _anchor_matrix = np.asarray(vectors, dtype=np.float32)
    return _anchor_matrix


def scope_score(question: str, embedder) -> float:
    vector = np.asarray(embedder.embed_query(question), dtype=np.float32)
    return float((_anchors(embedder) @ vector).max())


def guardrail_reason(question: str, embedder) -> str | None:
    """Return a short reason when the question must not be searched or sent to the LLM."""
    text = question.strip()
    if PROMPT_INJECTION.search(text):
        return "prompt_injection"
    if OTHER_BRANDS.search(text):
        return "other_brand"
    if scope_score(text, embedder) < SCOPE_THRESHOLD:
        return "out_of_scope"
    return None
