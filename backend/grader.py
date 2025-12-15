import json
import os
from typing import List, Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = "gpt-4.1-mini"


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _norm_page(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def grade_math_qa(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not qa_items:
        return []

    client = _client()

    lines = ["Here are some Year 5–8 maths questions and the student's answers:\n"]
    for item in qa_items:
        page = item.get("page_number")
        qn = item.get("question_number", "")
        qtext = item.get("question_text", "")
        sans = item.get("student_answer", "")
        work = item.get("student_working", "")

        if page is not None:
            lines.append(f"Question {qn} (page {page}): {qtext}")
        else:
            lines.append(f"Question {qn}: {qtext}")

        if work:
            lines.append(f"Student working: {work}")
        lines.append(f"Student final answer: {sans}")
        lines.append("")

    qa_block = "\n".join(lines)

    instructions = """
You are marking Year 5–8 maths short-answer questions.

You will receive multiple questions. Some questions include a page number like:
"Question 3 (page 2): ..."

For each question:
1. Work out the correct answer yourself.
2. Compare with the student's final answer.
3. Score:
   - 1 if mathematically correct (equivalent forms allowed, e.g. 1/2 = 0.5)
   - 0 if wrong, BLANK or UNREADABLE
4. Give short feedback.

Assume:
- max_score = 1 for every question.

Output ONLY JSON.
Return a JSON array of objects, one per question, with:
- page_number: integer page number if provided, otherwise null
- question_number: e.g. "1", "2(a)"
- score: 0 or 1
- max_score: always 1
- correct_answer: the correct answer you computed
- feedback: short feedback

Example:

[
  {
    "page_number": 2,
    "question_number": "3",
    "score": 1,
    "max_score": 1,
    "correct_answer": "21",
    "feedback": "Correct. 3 × 7 = 21."
  },
  {
    "page_number": null,
    "question_number": "1",
    "score": 0,
    "max_score": 1,
    "correct_answer": "0.4",
    "feedback": "Incorrect. 2/5 = 0.4."
  }
]
""".strip()

    resp = client.responses.create(
        model=TEXT_MODEL,
        instructions=instructions,
        input=qa_block,
        max_output_tokens=900,
    )

    raw = resp.output_text.strip()

    try:
        graded = json.loads(raw)
    except json.JSONDecodeError:
        print("Grader JSON parse failed. Raw output below:\n")
        print(raw)
        raise

    if not isinstance(graded, list):
        raise ValueError("Expected a JSON array from grader model")

    out: List[Dict[str, Any]] = []
    for item in graded:
        if not isinstance(item, dict):
            continue

        out.append(
            {
                "page_number": _norm_page(item.get("page_number")),
                "question_number": str(item.get("question_number", "")).strip(),
                "score": int(item.get("score", 0)),
                "max_score": int(item.get("max_score", 1)),
                "correct_answer": str(item.get("correct_answer", "")).strip(),
                "feedback": str(item.get("feedback", "")).strip(),
            }
        )

    return out


if __name__ == "__main__":
    fake_qa = [
        {
            "page_number": 1,
            "question_number": "1",
            "question_text": "Calculate 3 × 7.",
            "student_answer": "21",
            "student_working": "",
        },
        {
            "page_number": 2,
            "question_number": "1",
            "question_text": "What is 2/5 as a decimal?",
            "student_answer": "0.3",
            "student_working": "",
        },
    ]
    res = grade_math_qa(fake_qa)
    print(json.dumps(res, indent=2, ensure_ascii=False))
