"""Parse DealerListDetails.pdf into one record per dealer column.

Each visual column is an independent list. A state heading covers every
dealer below it in that column until the next state heading. The district
is read from that dealer's own address, so a state filter can match every
district underneath it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "DealerListDetails.pdf"
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "dealers.jsonl"
LOG_PATH = DATA_DIR / "dealer_parse_log.jsonl"

STATE_MAP = {
    "ANDHRA PRADESH": "Andhra Pradesh",
    "ANDAMAN": "Andaman and Nicobar Islands",
    "ANDAMAN AND NICOBAR ISLANDS": "Andaman and Nicobar Islands",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "CHANDIGARH": "Chandigarh",
    "CHATTISGARH": "Chhattisgarh",
    "CHHATTISGARH": "Chhattisgarh",
    "GOA": "Goa",
    "GUJARAT": "Gujarat",
    "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU & KASHMIR": "Jammu & Kashmir",
    "JAMMU AND KASHMIR": "Jammu & Kashmir",
    "J&K": "Jammu & Kashmir",
    "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "LADAKH": "Ladakh",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MADHYAPRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra",
    "MAHARASHTRA2": "Maharashtra",
    "MANIPUR": "Manipur",
    "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland",
    "NEW DELHI & NCR": "Delhi",
    "NEW DELHI": "Delhi",
    "DELHI": "Delhi",
    "NCR": "Delhi",
    "ORISSA": "Odisha",
    "ODISHA": "Odisha",
    "PONDICHERRY": "Puducherry",
    "PUDUCHERRY": "Puducherry",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "SIKKIM": "Sikkim",
    "TAMILNADU": "Tamil Nadu",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "TELENGANA": "Telangana",
    "TRIPURA": "Tripura",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UP": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "UTTARKHAND": "Uttarakhand",
    "WEST BENGAL": "West Bengal",
}

LANDMARK_RE = re.compile(
    r"\b(road|rd|opp|opposite|near|beside|nh|ward|circle|complex|nagar|"
    r"street|colony|office|stand|petrol|pump|hotel|bypass|by-pass|pass|"
    r"centre|center|market|highway|main|floor|plot|salai|theatre|theater|"
    r"hospital|station|bus|bank)\b",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"^(\d+)\s*[\.)]\s*(.+)$")
STATE_INDEX_RE = re.compile(r"^([A-Z]{1,3}\d*)\.?\s*(.+)$")
PINCODE_RE = re.compile(r"\b(\d{3}\s?\d{3})\b")
OFFICE_HEADER_RE = re.compile(r"^([NSEW]\d+)\s*[-–]\s*(.+)$", re.IGNORECASE)


def canonical_state(raw: str) -> str | None:
    cleaned = re.sub(r"\s+\d+$", "", raw.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).upper()
    cleaned = cleaned.replace(" AND ", " & ")
    if cleaned in STATE_MAP:
        return STATE_MAP[cleaned]
    compact = cleaned.replace(" ", "")
    if compact in STATE_MAP:
        return STATE_MAP[compact]
    return None


def parse_state_heading(text: str) -> str | None:
    direct = canonical_state(text)
    if direct and text.strip().upper() == text.strip() and not NAME_RE.match(text):
        # A bare state name. Indexed headings are handled below so that
        # "A. ANDHRA PRADESH" is not reduced to a stray letter plus junk.
        if STATE_INDEX_RE.match(text.strip()) and not text.strip().upper().startswith(
            tuple(STATE_MAP)
        ):
            pass
        elif text.strip().upper() in STATE_MAP or text.strip().upper().replace(" ", "") in STATE_MAP:
            return direct
    match = STATE_INDEX_RE.match(text.strip())
    if not match:
        return None
    return canonical_state(match.group(2))


def is_phone(text: str) -> bool:
    stripped = text.strip()
    digits = sum(character.isdigit() for character in stripped)
    if digits < 6:
        return False
    if re.match(r"(?i)^(ph|phone)\b", stripped):
        return True
    if stripped.startswith("+"):
        return True
    # A town followed by a pincode is an address, not a phone number.
    if re.search(r"[A-Za-z]{3,}.*\d{3}\s?\d{3}", stripped):
        return False
    letters = sum(character.isalpha() for character in stripped)
    if letters:
        return False
    return True


def is_district_heading(text: str) -> bool:
    if parse_state_heading(text) or is_phone(text) or NAME_RE.match(text):
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 4 or not letters.isupper():
        return False
    if any(character.isdigit() for character in text):
        return False
    words = text.split()
    if not 1 <= len(words) <= 8:
        return False
    if LANDMARK_RE.search(text):
        return False
    return True


def is_chrome(text: str, y: float, page_height: float) -> bool:
    if y > page_height - 36:
        return True
    lowered = text.strip().lower()
    if lowered in {"dealer directory", "studio stores", "oct-21"}:
        return True
    if re.fullmatch(r"(dealer directory|studio stores)\s+\d+", lowered):
        return True
    if re.fullmatch(r"oct-\d{2}", lowered):
        return True
    return False


def title_place(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,.-")
    if not value:
        return ""
    return value.title()


def district_from_address(address: str, hint: str = "") -> str:
    address = re.sub(r"\s+", " ", address).strip(" ,.-")
    if not address:
        return hint
    coded = re.match(r"^[A-Z]{1,3}\d+\.\s+([A-Za-z]+)", address)
    if coded:
        return title_place(coded.group(1)) or hint
    near = re.match(r"^(.+?)\s+near\s+", address, re.IGNORECASE)
    if near and not LANDMARK_RE.search(near.group(1)):
        return title_place(near.group(1)) or hint
    if "-" in address or "–" in address:
        left, right = re.split(r"\s*[-–]\s*", address, maxsplit=1)
        left, right = left.strip(), right.strip()
        left_landmark = bool(LANDMARK_RE.search(left))
        right_landmark = bool(LANDMARK_RE.search(right))
        if canonical_state(left) and right and not right_landmark:
            return title_place(right) or hint
        if left_landmark and not right_landmark and 0 < len(right.split()) <= 4:
            return title_place(right) or hint
        if not left_landmark and left:
            return title_place(left) or hint
        if not right_landmark and right:
            return title_place(right) or hint
        if left:
            return title_place(left) or hint
    if LANDMARK_RE.search(address) and len(address.split()) > 3:
        return hint
    if len(address) > 48:
        return hint
    return title_place(address) or hint


def split_phones(phone_lines: list[str]) -> tuple[str, str]:
    raw = " ".join(phone_lines)
    parts = re.split(r"\s*[;&]\s*|\s*,\s*(?=\+?\d)", raw)
    numbers = []
    for part in parts:
        part = part.strip(" ,;")
        if re.search(r"\d{5,}", part):
            numbers.append(part)
    if not numbers:
        return "", ""
    return numbers[0], "; ".join(numbers[1:])


def page_lines(page: fitz.Page) -> list[dict]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0 = min(span["bbox"][0] for span in line["spans"])
            y0 = min(span["bbox"][1] for span in line["spans"])
            lines.append({"text": text, "x": x0, "y": y0})
    lines.sort(key=lambda item: (item["y"], item["x"]))
    return lines


def column_anchors(xs: list[float]) -> list[float]:
    if not xs:
        return []
    ordered = sorted(xs)
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] > 70:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    centers = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= 2]
    if not centers:
        centers = [sum(cluster) / len(cluster) for cluster in clusters]
    return centers


def assign_columns(lines: list[dict]) -> list[list[dict]]:
    anchors = column_anchors([line["x"] for line in lines])
    columns = [[] for _ in anchors]
    for line in lines:
        index = min(range(len(anchors)), key=lambda item: abs(line["x"] - anchors[item]))
        if abs(line["x"] - anchors[index]) <= 90:
            columns[index].append(line)
    for column in columns:
        column.sort(key=lambda item: item["y"])
    return columns


def build_record(
    *,
    name: str,
    dealer_no: str,
    address_lines: list[str],
    phone_lines: list[str],
    state: str,
    district_hint: str,
    page_no: int,
    column_index: int,
    band: int,
    store_type: str,
) -> tuple[dict | None, dict | None]:
    address = re.sub(r"\s+", " ", " ".join(address_lines)).strip()
    phone, alt_phone = split_phones(phone_lines)
    if not state:
        return None, {
            "page": page_no,
            "column": column_index,
            "reason": "missing_state",
            "text": name,
        }
    district = district_from_address(address, district_hint)
    pincode_match = PINCODE_RE.search(address)
    pincode = re.sub(r"\s+", "", pincode_match.group(1)) if pincode_match else ""
    location = f"{state} -> {district}" if district else state
    kind = "Studio store" if store_type == "studio" else "Dealer"
    text_lines = [
        f"{kind}: {name}",
        f"State: {state}",
    ]
    if district:
        text_lines.append(f"District: {district}")
    if address:
        text_lines.append(f"Address: {address}")
    if phone:
        text_lines.append(f"Phone: {phone}")
    if alt_phone:
        text_lines.append(f"Alternate phone: {alt_phone}")
    metadata = {
        "source": "dealer",
        "record_type": store_type,
        "page": page_no,
        "column": column_index,
        "band": band,
        "dealer_name": name,
        "state": state,
        "dealer_no": dealer_no,
    }
    if district:
        metadata["district"] = district
        metadata["location"] = location
    if phone:
        metadata["phone"] = phone
    if alt_phone:
        metadata["alt_phone"] = alt_phone
    if pincode:
        metadata["pincode"] = pincode
    record = {
        "id": f"dealer-p{page_no}-c{column_index}-b{band}",
        "text": "\n".join(text_lines),
        "metadata": metadata,
    }
    warning = None
    if not phone:
        warning = {
            "page": page_no,
            "column": column_index,
            "reason": "missing_phone",
            "text": name,
        }
    return record, warning


def looks_like_unnumbered_name(lines: list[dict], index: int) -> bool:
    text = lines[index]["text"].strip()
    if not text or text.lower().startswith("total "):
        return False
    if any(character.isdigit() for character in text):
        return False
    if "-" in text or "–" in text or LANDMARK_RE.search(text):
        return False
    if len(text.split()) > 8:
        return False
    for upcoming in lines[index + 1 : index + 4]:
        upcoming_text = upcoming["text"]
        if is_phone(upcoming_text):
            return True
        if NAME_RE.match(upcoming_text) or parse_state_heading(upcoming_text):
            return False
    return False


def walk_column(
    lines: list[dict],
    *,
    state: str,
    district_hint: str,
    page_no: int,
    column_index: int,
    store_type: str,
) -> tuple[list[dict], list[dict], str, str]:
    records: list[dict] = []
    logs: list[dict] = []
    index = 0
    band = 0
    while index < len(lines):
        text = lines[index]["text"]
        heading = parse_state_heading(text)
        if heading:
            state = heading
            district_hint = ""
            index += 1
            continue
        if is_district_heading(text):
            district_hint = title_place(text)
            index += 1
            continue
        name_match = NAME_RE.match(text)
        unnumbered = name_match is None and looks_like_unnumbered_name(lines, index)
        if name_match is None and not unnumbered:
            if text.lower().startswith("total "):
                index += 1
                continue
            if not is_phone(text):
                logs.append(
                    {
                        "page": page_no,
                        "column": column_index,
                        "reason": "unaligned_line",
                        "text": text,
                    }
                )
            else:
                logs.append(
                    {
                        "page": page_no,
                        "column": column_index,
                        "reason": "phone_without_dealer",
                        "text": text,
                    }
                )
            index += 1
            continue
        if name_match:
            dealer_no, name = name_match.group(1), name_match.group(2).strip()
        else:
            dealer_no, name = "", text.strip()
        index += 1
        address_lines: list[str] = []
        while index < len(lines):
            nxt = lines[index]["text"]
            if (
                parse_state_heading(nxt)
                or is_district_heading(nxt)
                or NAME_RE.match(nxt)
                or is_phone(nxt)
            ):
                break
            address_lines.append(nxt)
            index += 1
        phone_lines: list[str] = []
        while index < len(lines) and is_phone(lines[index]["text"]):
            phone_lines.append(lines[index]["text"])
            index += 1
        band += 1
        record, warning = build_record(
            name=name,
            dealer_no=dealer_no,
            address_lines=address_lines,
            phone_lines=phone_lines,
            state=state,
            district_hint=district_hint,
            page_no=page_no,
            column_index=column_index,
            band=band,
            store_type=store_type,
        )
        if record:
            records.append(record)
        if warning:
            logs.append(warning)
    return records, logs, state, district_hint


def states_from_office_header(header: str) -> list[str]:
    body = OFFICE_HEADER_RE.match(header)
    raw = body.group(2) if body else header
    found = []
    for part in re.split(r",", raw):
        state = canonical_state(part)
        if state and state not in found:
            found.append(state)
    return found


def parse_regional_offices(page: fitz.Page, page_no: int) -> tuple[list[dict], list[dict]]:
    lines = [
        line
        for line in page_lines(page)
        if not is_chrome(line["text"], line["y"], page.rect.height)
        and line["text"].strip().lower() not in {"north", "south", "east", "west"}
        and line["text"].strip().lower() != "regional & area offices"
    ]
    ranges = [(0, 180), (180, 340), (340, 500), (500, 900)]
    records: list[dict] = []
    logs: list[dict] = []
    for column_index, (left, right) in enumerate(ranges):
        column = [line for line in lines if left <= line["x"] < right]
        column.sort(key=lambda item: item["y"])
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in column:
            if OFFICE_HEADER_RE.match(line["text"]):
                if current:
                    blocks.append(current)
                current = [line["text"]]
            elif line["text"].strip().upper() == "ROYAL ENFIELD SUPPORT":
                if current:
                    blocks.append(current)
                current = [line["text"]]
            else:
                current.append(line["text"])
        if current:
            blocks.append(current)
        for band, block in enumerate(blocks, start=1):
            header = block[0]
            body = [part.strip() for part in block[1:] if part.strip()]
            support = header.strip().upper() == "ROYAL ENFIELD SUPPORT"
            phone_lines = [part for part in body if is_phone(part) or part.lower().startswith("ph")]
            address_lines = [part for part in body if part not in phone_lines]
            address = re.sub(r"\s+", " ", " ".join(address_lines)).strip()
            phone, alt_phone = split_phones(phone_lines)
            if support:
                text = "Royal Enfield support\nPhone: 1800-210-0007"
                if phone:
                    text = f"Royal Enfield support\nPhone: {phone}"
                    if alt_phone:
                        text += f"\nAlternate phone: {alt_phone}"
                records.append(
                    {
                        "id": f"support-p{page_no}",
                        "text": text,
                        "metadata": {
                            "source": "dealer",
                            "record_type": "support",
                            "page": page_no,
                            "column": column_index,
                            "dealer_name": "Royal Enfield support",
                            "phone": phone or "1800-210-0007",
                        },
                    }
                )
                continue
            states = states_from_office_header(header)
            for part in address_lines:
                state = canonical_state(part.strip("()."))
                if state and state not in states:
                    states.append(state)
                paren = re.search(r"\(([A-Za-z &]+)\)", part)
                if paren:
                    state = canonical_state(paren.group(1))
                    if state and state not in states:
                        states.append(state)
            if not states:
                logs.append(
                    {
                        "page": page_no,
                        "column": column_index,
                        "reason": "office_missing_state",
                        "text": header,
                    }
                )
                continue
            district = ""
            city_match = re.search(
                r"([A-Za-z]+)\s*[-–]\s*\d{3}\s?\d{3}",
                address,
            )
            if city_match:
                district = title_place(city_match.group(1))
                if canonical_state(district):
                    district = ""
            for state in states:
                metadata = {
                    "source": "dealer",
                    "record_type": "regional_office",
                    "page": page_no,
                    "column": column_index,
                    "band": band,
                    "dealer_name": f"Royal Enfield regional office {header}",
                    "state": state,
                }
                if district:
                    metadata["district"] = district
                    metadata["location"] = f"{state} -> {district}"
                if phone:
                    metadata["phone"] = phone
                if alt_phone:
                    metadata["alt_phone"] = alt_phone
                pincode_match = PINCODE_RE.search(address)
                if pincode_match:
                    metadata["pincode"] = re.sub(r"\s+", "", pincode_match.group(1))
                text_lines = [
                    f"Regional office: {header}",
                    f"State: {state}",
                    "Covers: " + ", ".join(states),
                ]
                if district:
                    text_lines.append(f"District: {district}")
                if address:
                    text_lines.append(f"Address: {address}")
                if phone:
                    text_lines.append(f"Phone: {phone}")
                if alt_phone:
                    text_lines.append(f"Alternate phone: {alt_phone}")
                state_key = re.sub(r"[^a-z0-9]+", "-", state.lower()).strip("-")
                records.append(
                    {
                        "id": f"office-p{page_no}-c{column_index}-b{band}-{state_key}",
                        "text": "\n".join(text_lines),
                        "metadata": metadata,
                    }
                )
    return records, logs


def parse_pdf(path: Path = PDF_PATH) -> tuple[list[dict], list[dict]]:
    document = fitz.open(path)
    records: list[dict] = []
    logs: list[dict] = []
    carried_state = ["", "", ""]
    carried_district = ["", "", ""]
    store_type = "dealer"
    for page_index, page in enumerate(document):
        page_no = page_index + 1
        raw_lines = page_lines(page)
        if any(line["text"].strip().lower() == "studio stores" for line in raw_lines):
            store_type = "studio"
        if any(line["text"].strip().lower() == "regional & area offices" for line in raw_lines):
            office_records, office_logs = parse_regional_offices(page, page_no)
            records.extend(office_records)
            logs.extend(office_logs)
            continue
        body = [
            line
            for line in raw_lines
            if not is_chrome(line["text"], line["y"], page.rect.height)
            and line["text"].strip().lower() not in {"dealer directory", "studio stores"}
        ]
        columns = assign_columns(body)
        for column_index, column in enumerate(columns):
            state = carried_state[column_index] if column_index < len(carried_state) else ""
            district_hint = (
                carried_district[column_index] if column_index < len(carried_district) else ""
            )
            column_records, column_logs, state, district_hint = walk_column(
                column,
                state=state,
                district_hint=district_hint,
                page_no=page_no,
                column_index=column_index,
                store_type=store_type,
            )
            records.extend(column_records)
            logs.extend(column_logs)
            if column_index >= len(carried_state):
                carried_state.append(state)
                carried_district.append(district_hint)
            else:
                carried_state[column_index] = state
                carried_district[column_index] = district_hint
    return records, logs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    records, logs = parse_pdf()
    write_jsonl(OUTPUT_PATH, records)
    write_jsonl(LOG_PATH, logs)
    states = sorted({row["metadata"]["state"] for row in records if row["metadata"].get("state")})
    print(f"dealers {len(records)} log {len(logs)} states {len(states)}")
    print(", ".join(states))


if __name__ == "__main__":
    main()
