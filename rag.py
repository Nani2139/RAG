"""Retrieve with a metadata filter first, then cosine similarity, then Groq.

The answer is limited to the retrieved chunks. An empty or weak retrieval
does not call the model. Questions that ask how many dealers are in a state
or district are counted from the metadata filter, not from the top matches.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from guardrail import OTHER_BRANDS, OUT_OF_SCOPE, PROMPT_INJECTION, guardrail_reason

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "royalenfield"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Cosine distance in Chroma is 1 - cosine similarity. Hits farther than this
# are treated as unrelated to the question.
DISTANCE_CUTOFF = 0.72
REFUSAL = "The Royal Enfield documents do not contain the answer to that question."
DEALERS_PATH = ROOT / "data" / "dealers.jsonl"
COUNT_QUESTION = re.compile(
    r"\b(how many|number of|count of|total number of|total)\b",
    re.IGNORECASE,
)
DEALER_WORD = re.compile(r"\b(dealers?|showrooms?|outlets?|studio stores?)\b", re.IGNORECASE)
DEALER_INTENT = re.compile(
    r"\b(dealers?|showrooms?|outlets?|studio stores?|phones?|addresses?|contacts?)\b",
    re.IGNORECASE,
)
SPEC_INTENT = re.compile(
    r"\b(seat height|mileage|price|prices|engine|torque|power|kerb|colour|color|"
    r"brakes?|tyres?|tires?|fuel tank|gearbox|ground clearance|wheelbase|specification|specs?)\b",
    re.IGNORECASE,
)
BROCHURE_PATH = ROOT / "data" / "brochure.jsonl"
STATE_ALIASES = {
    "andhra": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "arunachal": "Arunachal Pradesh",
    "arunachalpradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal": "Himachal Pradesh",
    "himachalpradesh": "Himachal Pradesh",
    "jammu": "Jammu & Kashmir",
    "kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "jk": "Jammu & Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "mp": "Madhya Pradesh",
    "madhyapradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "delhi": "Delhi",
    "ncr": "Delhi",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tn": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "telangana": "Telangana",
    "telengana": "Telangana",
    "tripura": "Tripura",
    "up": "Uttar Pradesh",
    "uttarpradesh": "Uttar Pradesh",
    "uk": "Uttarakhand",
    "uttarakhand": "Uttarakhand",
    "uttarkhand": "Uttarakhand",
    "westbengal": "West Bengal",
    "bengal": "West Bengal",
    "andaman": "Andaman and Nicobar Islands",
}

SYSTEM_PROMPT = """You answer questions about Royal Enfield using only the context below.
Copy dealer names, addresses, phone numbers, and specifications exactly as written.
When a dealer is listed, keep that dealer's name with that dealer's own address and phone numbers.
If the context does not contain the answer, reply exactly: {refusal}
Do not add facts from outside the context.

Context:
{context}"""


_embedder = None


def embeddings():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def vector_store(embedder=None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder or embeddings(),
    )


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9&]", "", value.lower())


def _plain(value: str) -> str:
    text = re.sub(r"[^a-z0-9& ]", " ", value.lower())
    return re.sub(r"\s+", " ", text).strip()


_place_index: dict | None = None


def place_index() -> dict:
    global _place_index
    if _place_index is not None:
        return _place_index
    states = set()
    district_states: dict[str, set[str]] = {}
    if DEALERS_PATH.exists():
        with DEALERS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                metadata = json.loads(line).get("metadata", {})
                state = metadata.get("state") or ""
                district = metadata.get("district") or ""
                if state:
                    states.add(state)
                if district and state:
                    district_states.setdefault(district, set()).add(state)
    _place_index = {"states": states, "district_states": district_states}
    return _place_index


def _phrase_in(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def places_in_question(question: str) -> tuple[str, str]:
    """Read a state and district named in the question. District wins when both appear."""
    text = _plain(question)
    compact = _compact(question)
    index = place_index()
    district = ""
    district_state = ""
    for name in sorted(index["district_states"], key=len, reverse=True):
        if _phrase_in(text, _plain(name)):
            district = name
            states = index["district_states"][name]
            if len(states) == 1:
                district_state = next(iter(states))
            break
    state = ""
    for name in sorted(index["states"], key=len, reverse=True):
        if _phrase_in(text, _plain(name)):
            state = name
            break
    if not state:
        for alias, canonical in sorted(STATE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if _phrase_in(text, alias) or compact == _compact(alias):
                state = canonical
                break
            if len(alias) >= 4 and _phrase_in(compact, _compact(alias)):
                state = canonical
                break
    if district and not state:
        state = district_state
    return state, district


def is_dealer_count_question(question: str) -> bool:
    return bool(COUNT_QUESTION.search(question) and DEALER_WORD.search(question))


def count_dealers(state: str, district: str) -> dict[str, int]:
    """Count every dealer and studio-store row for a place. Similarity search only returns a few chunks, so a total has to come from the directory itself."""
    counts = {"dealer": 0, "studio": 0}
    if not DEALERS_PATH.exists():
        return counts
    with DEALERS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metadata = json.loads(line).get("metadata") or {}
            if metadata.get("source") != "dealer":
                continue
            kind = metadata.get("record_type")
            if kind not in counts:
                continue
            if state and metadata.get("state") != state:
                continue
            if district and metadata.get("district") != district:
                continue
            counts[kind] += 1
    return counts


def format_count(state: str, district: str, counts: dict[str, int]) -> str:
    dealers = counts["dealer"]
    studios = counts["studio"]
    total = dealers + studios
    if district and state:
        place = f"{district}, {state}"
    else:
        place = district or state or "the directory"
    if total == 0:
        return f"The directory does not list any Royal Enfield dealers in {place}."
    if dealers and studios:
        return (
            f"{place} has {total} Royal Enfield dealer listings: "
            f"{dealers} in the dealer directory and {studios} studio stores."
        )
    if studios and not dealers:
        return f"{place} has {studios} Royal Enfield studio stores."
    noun = "dealer" if dealers == 1 else "dealers"
    return f"{place} has {dealers} Royal Enfield {noun}."


_brochure_titles: list[str] | None = None


def brochure_titles() -> list[str]:
    global _brochure_titles
    if _brochure_titles is not None:
        return _brochure_titles
    titles: list[str] = []
    if BROCHURE_PATH.exists():
        with BROCHURE_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                title = (json.loads(line).get("metadata") or {}).get("title") or ""
                if title and not title.lower().startswith("royal enfield india") and title not in titles:
                    titles.append(title)
    _brochure_titles = titles
    return titles


def models_in_question(question: str) -> list[str]:
    """Model names written in the question, longest title first so Classic 350 is not read as Classic 650."""
    text = _plain(question)
    found: list[str] = []
    for title in sorted(brochure_titles(), key=len, reverse=True):
        phrase = _plain(title)
        if phrase and _phrase_in(text, phrase):
            found.append(title)
            text = re.sub(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", " ", text)
    return found


def query_filter(question: str) -> dict | None:
    """Metadata filter taken from the question, applied before cosine similarity."""
    state, district = places_in_question(question)
    models = models_in_question(question)
    dealerish = DEALER_INTENT.search(question) is not None
    specish = SPEC_INTENT.search(question) is not None or bool(models)
    if dealerish or ((state or district) and not specish):
        return build_filter("dealer", state, district)
    if len(models) == 1:
        return {
            "$and": [
                {"source": {"$eq": "brochure"}},
                {"title": {"$eq": models[0]}},
            ]
        }
    if specish:
        return {"source": {"$eq": "brochure"}}
    return None


def build_filter(source: str, state: str, district: str) -> dict | None:
    clauses = []
    if source == "brochure":
        clauses.append({"source": {"$eq": "brochure"}})
    elif source == "dealer" or state or district:
        clauses.append({"source": {"$eq": "dealer"}})
    if state:
        clauses.append({"state": {"$eq": state}})
    if district:
        clauses.append({"district": {"$eq": district}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def format_context(docs) -> str:
    blocks = []
    for document in docs:
        meta = document.metadata
        if meta.get("source") == "brochure":
            label = f"Brochure page {meta.get('page')}"
            if meta.get("title"):
                label += f" — {meta['title']}"
        else:
            label = meta.get("dealer_name") or "Dealer"
            if meta.get("location"):
                label += f" ({meta['location']})"
            elif meta.get("state"):
                label += f" ({meta['state']})"
            label += f", page {meta.get('page')}"
        blocks.append(f"[{label}]\n{document.page_content}")
    return "\n\n".join(blocks)


def source_view(document, distance: float) -> dict:
    meta = document.metadata
    return {
        "source": meta.get("source", ""),
        "page": meta.get("page"),
        "title": meta.get("title") or meta.get("dealer_name") or "",
        "state": meta.get("state", ""),
        "district": meta.get("district", ""),
        "location": meta.get("location") or meta.get("state") or "",
        "distance": round(float(distance), 4),
        "text": document.page_content,
    }


def retrieve(question: str, store: Chroma | None = None):
    store = store or vector_store()
    where = query_filter(question)
    kwargs = {"k": 12}
    if where:
        kwargs["filter"] = where
    hits = store.similarity_search_with_score(question, **kwargs)
    kept = [(document, distance) for document, distance in hits if distance <= DISTANCE_CUTOFF]
    return dedupe(kept)[:4]


def dedupe(hits):
    best = {}
    order = []
    for document, distance in hits:
        meta = document.metadata
        key = (
            meta.get("source"),
            meta.get("title") or meta.get("dealer_name"),
            meta.get("page"),
            meta.get("district", ""),
            meta.get("phone", ""),
        )
        if key not in best:
            order.append(key)
            best[key] = (document, distance)
        elif distance < best[key][1]:
            best[key] = (document, distance)
    return [best[key] for key in order]


def answer_question(question: str, store: Chroma | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    question = question.strip()
    if not question:
        return {"answer": "Enter a question.", "sources": []}
    if is_dealer_count_question(question):
        if PROMPT_INJECTION.search(question) or OTHER_BRANDS.search(question):
            return {"answer": OUT_OF_SCOPE, "sources": [], "guardrail": "out_of_scope"}
        count_state, count_district = places_in_question(question)
        counts = count_dealers(count_state, count_district)
        return {
            "answer": format_count(count_state, count_district, counts),
            "sources": [],
            "guardrail": "dealer_count",
        }
    embedder = embeddings()
    reason = guardrail_reason(question, embedder)
    if reason:
        return {"answer": OUT_OF_SCOPE, "sources": [], "guardrail": reason}
    hits = retrieve(question, store or vector_store(embedder))
    sources = [source_view(document, distance) for document, distance in hits]
    if not hits:
        return {"answer": REFUSAL, "sources": []}
    context = format_context([document for document, _distance in hits])
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "A Groq API key is required to write the answer. Add GROQ_API_KEY to the .env file. The matching document excerpts are listed below.",
            "sources": sources,
        }
    chat = ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = prompt | chat | StrOutputParser()
    text = chain.invoke({"question": question, "context": context, "refusal": REFUSAL})
    return {"answer": text.strip(), "sources": sources}
