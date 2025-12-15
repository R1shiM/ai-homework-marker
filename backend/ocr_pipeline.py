import base64
import json
import os
import re
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL = "gpt-4.1-mini"


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _to_data_url_bytes(b: bytes, mime: str) -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _guess_mime_from_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def _image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return _to_data_url_bytes(b, _guess_mime_from_path(path))


def _pdf_to_page_data_urls(pdf_path: str, zoom: float = 2.0) -> List[str]:
    doc = fitz.open(pdf_path)
    out: List[str] = []

    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        out.append(_to_data_url_bytes(png_bytes, "image/png"))

    return out


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_loose(raw: str) -> Any:
    raw = _strip_code_fences(raw)

    start_idx = None
    for i, ch in enumerate(raw):
        if ch in "{[":
            start_idx = i
            break
    if start_idx is None:
        raise json.JSONDecodeError("No JSON object/array start found", raw, 0)

    candidate = raw[start_idx:]
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(candidate)
    return obj


def _extract_qa_from_one_page(
    client: OpenAI,
    image_data_url: str,
    page_number: Optional[int],
) -> List[Dict[str, Any]]:
    prompt = """
I have a Year 5–8 maths worksheet in the image.

The page has:
- numbered questions
- handwritten student answers on the same page

I want a clean JSON list of question/answer pairs.

For each numbered question, extract:
- question_number: like "1", "2(a)" etc
- question_text: full question as normal text
- student_answer: student's final answer (even if wrong)
- student_working: any visible working/steps

Rules:
- Ignore headings, name fields, teacher notes.
- If answer is unreadable -> student_answer = "UNREADABLE"
- If there is clearly no answer -> student_answer = "BLANK"

Output:
Return ONLY a JSON array, like:

[
  {
    "question_number": "1",
    "question_text": "...",
    "student_answer": "...",
    "student_working": "..."
  }
]
""".strip()

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        max_output_tokens=900,
    )

    raw = resp.output_text.strip()

    try:
        data = _parse_json_loose(raw)
    except json.JSONDecodeError:
        print("Vision model JSON parse failed. Raw output below:\n")
        print(raw)
        raise

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array from vision model")

    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        qn = str(item.get("question_number", "")).strip()
        qtext = str(item.get("question_text", "")).strip()
        sans = str(item.get("student_answer", "")).strip()
        work = str(item.get("student_working", "")).strip()

        if not qn and not qtext:
            continue

        entry: Dict[str, Any] = {
            "question_number": qn,
            "question_text": qtext,
            "student_answer": sans,
            "student_working": work,
        }

        if page_number is not None:
            entry["page_number"] = page_number

        cleaned.append(entry)

    return cleaned


def extract_qa_from_file(path: str) -> List[Dict[str, Any]]:
    client = _client()
    ext = os.path.splitext(path.lower())[1]

    if ext == ".pdf":
        page_urls = _pdf_to_page_data_urls(path, zoom=2.0)
        all_items: List[Dict[str, Any]] = []

        for idx, page_url in enumerate(page_urls, start=1):
            page_items = _extract_qa_from_one_page(client, page_url, page_number=idx)
            all_items.extend(page_items)

        return all_items

    image_url = _image_file_to_data_url(path)
    return _extract_qa_from_one_page(client, image_url, page_number=None)


if __name__ == "__main__":
    sample_img = os.path.join("..", "sample_data", "worksheet1.jpg")
    sample_pdf = os.path.join("..", "sample_data", "worksheet1.pdf")

    path = sample_pdf if os.path.exists(sample_pdf) else sample_img
    qa = extract_qa_from_file(path)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
