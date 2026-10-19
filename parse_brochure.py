"""Parse broucher.pdf into one record per motorcycle.

The file is text and tables, with no images. A model's specification table
can continue onto the next page, so the chunk is the model, not the page.
Each table row stays a single "field: value" line on that model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "broucher.pdf"
OUTPUT_PATH = ROOT / "data" / "brochure.jsonl"

MODELS = [
    "Super Meteor 650",
    "Continental GT 650",
    "Goan Classic 350",
    "Interceptor 650",
    "Himalayan 450",
    "Himalayan 440",
    "Classic 350",
    "Classic 650",
    "Bullet 350",
    "Bullet 650",
    "Hunter 350",
    "Meteor 350",
    "Scram 440",
    "Guerrilla 450",
    "Shotgun 650",
    "Bear 650",
]
MODEL_SET = set(MODELS)


def clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def line_outside_tables(block: dict, tables: list[fitz.Rect]) -> list[tuple[float, str]]:
    kept = []
    for line in block.get("lines", []):
        text = clean("".join(span["text"] for span in line["spans"]))
        if not text:
            continue
        y0 = min(span["bbox"][1] for span in line["spans"])
        y1 = max(span["bbox"][3] for span in line["spans"])
        midpoint = (y0 + y1) / 2
        if any(table.y0 - 1 <= midpoint <= table.y1 + 1 for table in tables):
            continue
        kept.append((y0, text))
    return kept


def append_table(lines: list[str], rows: list[list[str]]) -> None:
    header = [clean(cell).lower() for cell in rows[0]] if rows else []
    body = rows[1:] if header and header[0] == "model" else rows
    if header and header[0] == "model":
        labels = [clean(cell) for cell in rows[0][1:]]
        for row in body:
            cells = [clean(cell) for cell in row]
            if not any(cells):
                continue
            model = cells[0]
            details = []
            for label, value in zip(labels, cells[1:]):
                if label and value:
                    details.append(f"{label}: {value}")
            if model:
                lines.append(f"{model} — " + "; ".join(details))
        return
    for row in rows:
        cells = [clean(cell) for cell in row]
        if not any(cells):
            continue
        label = cells[0] if cells else ""
        value = " ".join(cell for cell in cells[1:] if cell)
        if not label and value and lines:
            lines[-1] = f"{lines[-1]} {value}".strip()
            continue
        if label and value:
            lines.append(f"{label}: {value}")
        elif label or value:
            lines.append(label or value)


def page_items(page: fitz.Page) -> list[tuple[float, str, object]]:
    found = page.find_tables()
    tables = list(found.tables) if found else []
    rects = [fitz.Rect(table.bbox) for table in tables]
    items: list[tuple[float, str, object]] = []
    for table, rect in zip(tables, rects):
        rows = []
        for row in table.extract() or []:
            cells = [cell or "" for cell in row]
            if any(clean(cell) for cell in cells):
                rows.append(cells)
        if rows:
            items.append((rect.y0, "table", rows))
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for y0, text in line_outside_tables(block, rects):
            items.append((y0, "text", text))
    items.sort(key=lambda item: item[0])
    return items


def heading_in(text: str) -> str | None:
    if text in MODEL_SET:
        return text
    for model in MODELS:
        if text == model or text.startswith(model + " "):
            return model
    return None


def parse_pdf(path: Path = PDF_PATH) -> list[dict]:
    document = fitz.open(path)
    overview_lines: list[str] = []
    overview_pages: set[int] = set()
    sections: list[dict] = []
    current: dict | None = None

    def begin(title: str, page_no: int, remainder: str = "") -> None:
        nonlocal current
        current = {"title": title, "pages": {page_no}, "lines": []}
        sections.append(current)
        if remainder:
            current["lines"].append(remainder)

    for index, page in enumerate(document):
        page_no = index + 1
        items = page_items(page)
        page_text = clean(page.get_text("text"))
        is_overview = "Quick reference" in page_text and page_no == 1
        for _y, kind, payload in items:
            if is_overview:
                overview_pages.add(page_no)
                if kind == "text":
                    overview_lines.append(str(payload))
                else:
                    append_table(overview_lines, payload)  # type: ignore[arg-type]
                continue
            if kind == "text":
                text = str(payload)
                model = heading_in(text)
                if model:
                    remainder = clean(text[len(model) :])
                    begin(model, page_no, remainder)
                    continue
                if current is None:
                    continue
                current["pages"].add(page_no)
                current["lines"].append(text)
                continue
            if current is None:
                continue
            current["pages"].add(page_no)
            append_table(current["lines"], payload)  # type: ignore[arg-type]

    records = []
    if overview_lines:
        pages = sorted(overview_pages) or [1]
        records.append(
            {
                "id": "brochure-quick-reference",
                "text": "\n".join(overview_lines),
                "metadata": {
                    "source": "brochure",
                    "page": pages[0],
                    "title": "Royal Enfield India motorcycle range",
                },
            }
        )
    for section in sections:
        pages = sorted(section["pages"])
        body = "\n".join(line for line in section["lines"] if line.strip())
        text = f"Model: {section['title']}\n{body}".strip()
        records.append(
            {
                "id": "brochure-" + re.sub(r"[^a-z0-9]+", "-", section["title"].lower()).strip("-"),
                "text": text,
                "metadata": {
                    "source": "brochure",
                    "page": pages[0],
                    "title": section["title"],
                },
            }
        )
    return records


def main() -> None:
    records = parse_pdf()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} brochure records")
    for record in records:
        print(f"{record['metadata']['page']:2} {record['metadata']['title']}: {len(record['text'])} chars")


if __name__ == "__main__":
    main()
