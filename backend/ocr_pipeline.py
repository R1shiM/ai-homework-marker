import base64
import json
import os
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Vision-capable model for OCR/extraction
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1-mini")


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


# JSOn helpers
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    return re.sub(_CODE_FENCE_RE, "", s).strip()


def _remove_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)


def _extract_first_json_blob(s: str) -> str:
    """
    Find the first JSON object/array in a string. Handles outputs like:
    ```json
    [...]
    ```
    or extra preamble text.
    """
    s = _strip_code_fences(s)

    # Try direct first
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # Find first { or [
    start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not start_candidates:
        raise json.JSONDecodeError("No JSON object/array found", s, 0)
    start = min(start_candidates)

    # Find last } or ]
    end_candidates = [i for i in (s.rfind("}"), s.rfind("]")) if i != -1]
    if not end_candidates:
        raise json.JSONDecodeError("No JSON object/array end found", s, start)
    end = max(end_candidates)

    blob = s[start : end + 1].strip()
    blob = _remove_trailing_commas(blob)
    return blob


def _loads_loose(raw: str) -> Any:
    blob = _extract_first_json_blob(raw)
    return json.loads(blob)


# utilities
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
    """
    Converts each PDF page into a PNG data URL using PyMuPDF.

    NOTE:
    - Requires: pip install pymupdf
    - Import name is `fitz`
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF is not installed or not importable. Install with:\n"
            "  pip install pymupdf\n"
            f"Original error: {e}"
        )

    doc = fitz.open(pdf_path)
    out: List[str] = []
    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        out.append(_to_data_url_bytes(png_bytes, "image/png"))

    return out


# OCR extraction
def _extract_qa_from_one_page(
    client: OpenAI,
    image_data_url: str,
    page_number: Optional[int],
) -> List[Dict[str, Any]]:
    prompt = """
You are extracting a Year 5–8 maths worksheet from an image.

The page contains:
- numbered questions
- handwritten student answers on the same page

Return a clean JSON array of question/answer items.

For each item, extract:
- question_number: like "1", "2(a)", "3(b)" etc
- question_text: full question as normal text
- student_answer: student's final answer (even if wrong)
- student_working: any visible working/steps

Rules:
- Ignore headings, names, teacher notes, decorations.
- If answer is unreadable -> student_answer = "UNREADABLE"
- If there is clearly no answer -> student_answer = "BLANK"

IMPORTANT (multi-problem questions):
- If a single numbered question contains multiple separate problems (like a list of sums),
  split them into subparts: "2(a)", "2(b)", "2(c)"... even if the worksheet does not label them.
  Each subpart should have its own question_text and student_answer.

Output:
Return ONLY a JSON array. Do not wrap in markdown code fences.
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
        max_output_tokens=1200,
    )

    raw = resp.output_text.strip()

    try:
        data = _loads_loose(raw)
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

    # image file
    image_url = _image_file_to_data_url(path)
    return _extract_qa_from_one_page(client, image_url, page_number=None)


if __name__ == "__main__":
    sample_img = os.path.join("..", "sample_data", "worksheet1.jpg")
    sample_pdf = os.path.join("..", "sample_data", "worksheet1.pdf")

    path = sample_pdf if os.path.exists(sample_pdf) else sample_img
    qa = extract_qa_from_file(path)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
