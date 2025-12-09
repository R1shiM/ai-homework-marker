import base64
import json
import os
from typing import List, Dict, Any

from openai import OpenAI

VISION_MODEL = "gpt-4.1-mini"


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_qa_from_image(image_path: str) -> List[Dict[str, Any]]:
    """
    Given one worksheet image (questions + student answers),
    return a list of dicts with question + student's answer.
    """
    client = _client()
    image_data_url = _image_to_data_url(image_path)

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
        max_output_tokens=800,
    )

    raw = resp.output_text.strip()

    try:
        data = json.loads(raw)
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

        cleaned.append(
            {
                "question_number": qn,
                "question_text": qtext,
                "student_answer": sans,
                "student_working": work,
            }
        )

    return cleaned


if __name__ == "__main__":
    # quick local poke
    sample = os.path.join("..", "sample_data", "worksheet1.jpg")
    qa = extract_qa_from_image(sample)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
