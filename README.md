# Royal Enfield assistant

A local chat assistant for Royal Enfield owners. Answers come only from two documents in this folder:

- `broucher.pdf` — model specifications and prices
- `DealerListDetails.pdf` — dealer and studio-store contacts

Questions outside that scope get a fixed reply. The model is not asked to fill in missing facts.

## How a question is answered

1. Unrelated questions, other motorcycle brands, and prompt-injection attempts are stopped before any search.
2. The question text sets the metadata filter. A state or district name limits the search to dealers in that place. A model name limits it to that model's brochure pages. A dealer question with no place searches the dealer directory. A specification question with no single model searches the brochure.
3. Cosine similarity runs only on the chunks that pass that filter.
4. Groq writes the answer from those chunks. If nothing relevant is retrieved, the assistant says the documents do not contain the answer.

"How many dealers" questions are not answered from the nearest chunks. The place is read from the question and every matching dealer and studio store in `data/dealers.jsonl` is counted.

## Setup

Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Groq API key in `.env`:

```
GROQ_API_KEY=your-key-here
GROQ_MODEL=openai/gpt-oss-120b
```

`.env` is gitignored. Do not commit it.

## Build the index

The parsed JSONL files are already in `data/`. Rebuild them only if a PDF changes.

```bash
python parse_dealers.py
python parse_brochure.py
python ingest.py
```

`ingest.py` embeds the records with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a local Chroma collection at `chroma_db/`. That folder is gitignored, so run ingest once after cloning.

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints. Ask in plain language, for example:

- what is the seat height of the Classic 350
- how many dealers in Andhra
- how many dealers in Kurnool

There are no source, state, or district controls. The question itself decides the filter.
