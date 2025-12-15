import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = "gpt-4.1-mini"

BATCH_MAX_ITEMS = 6
BATCH_MAX_CHARS = 2600
RETRIES = 3

BLANK_TOKENS = {"", "BLANK", "UNREADABLE"}


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


def _key(page: Optional[int], qn: str) -> Tuple[Optional[int], str]:
    return (page, qn.strip())


def _estimate_chars(items: List[Dict[str, Any]]) -> int:
    total = 0
    for it in items:
        total += len(str(it.get("question_text", "")))
        total += len(str(it.get("student_answer", "")))
        total += len(str(it.get("student_working", "")))
    return total


def _should_batch_page(items: List[Dict[str, Any]]) -> bool:
    if len(items) == 0:
        return False
    if len(items) > BATCH_MAX_ITEMS:
        return False
    if _estimate_chars(items) > BATCH_MAX_CHARS:
        return False

    qnums = [str(it.get("question_number", "")).strip() for it in items]
    if any(not q for q in qnums):
        return False
    if any(c > 1 for c in Counter(qnums).values()):
        return False

    return True


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


_NUM_RE = re.compile(r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\$?\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    return s


def _extract_numbers(s: str) -> List[str]:
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    nums = _NUM_RE.findall(s)
    return [n.strip() for n in nums]


def _to_float_token(tok: str) -> Optional[float]:
    tok = tok.strip()
    tok = tok.replace("$", "").replace(",", "")
    if tok in {"", "-", "+", "."}:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _numeric_equivalent(a: str, b: str, tol: float = 1e-9) -> bool:
    af = _to_float_token(a)
    bf = _to_float_token(b)
    if af is None or bf is None:
        return False
    return abs(af - bf) <= tol


def _numbers_equivalent(student: str, correct: str) -> Optional[bool]:
    s_nums = _extract_numbers(student)
    c_nums = _extract_numbers(correct)

    if not s_nums or not c_nums:
        return None

    # single number vs single number
    if len(s_nums) == 1 and len(c_nums) == 1:
        return _numeric_equivalent(s_nums[0], c_nums[0])

    # list/tuple-y answers: compare numeric sequences exactly
    s_vals = [_to_float_token(x) for x in s_nums]
    c_vals = [_to_float_token(x) for x in c_nums]
    if any(v is None for v in s_vals) or any(v is None for v in c_vals):
        return None

    if len(s_vals) != len(c_vals):
        return False

    for sv, cv in zip(s_vals, c_vals):
        if abs(sv - cv) > 1e-9:
            return False
    return True


def _solve_one(client: OpenAI, qa: Dict[str, Any], attempts: int = RETRIES) -> Dict[str, Any]:
    page = _norm_page(qa.get("page_number"))
    qn = str(qa.get("question_number", "")).strip()
    qtext = str(qa.get("question_text", "")).strip()

    prompt = "\n".join(
        [
            f"PAGE: {page if page is not None else 'null'}",
            f"QID: {qn}",
            f"Question: {qtext}",
        ]
    )

    instructions = """
Solve ONE Year 5–8 maths question.

Return ONLY raw JSON (no markdown) as one object:
{
  "page_number": <integer or null>,
  "question_number": "<exact QID>",
  "correct_answer": "<final answer only, concise>"
}

Don't include working unless it's required to define the final answer (e.g., list of pairs).
""".strip()

    last_raw = ""
    for _ in range(attempts):
        resp = client.responses.create(
            model=TEXT_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=350,
        )
        raw = resp.output_text.strip()
        last_raw = raw

        try:
            obj = _parse_json_loose(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        return {
            "page_number": _norm_page(obj.get("page_number", page)),
            "question_number": str(obj.get("question_number", qn)).strip() or qn,
            "correct_answer": str(obj.get("correct_answer", "")).strip(),
        }

    print("Solve parse failed after retries (single). Raw output below:\n")
    print(last_raw)

    return {
        "page_number": page,
        "question_number": qn,
        "correct_answer": "",
    }


def _solve_page_batch(
    client: OpenAI,
    page: Optional[int],
    items: List[Dict[str, Any]],
    attempts: int = RETRIES,
) -> Optional[List[Dict[str, Any]]]:
    blocks = [f"PAGE: {page if page is not None else 'null'}", ""]

    for it in items:
        qn = str(it.get("question_number", "")).strip()
        qtext = str(it.get("question_text", "")).strip()
        blocks.append(f"QID: {qn}")
        blocks.append(f"Question: {qtext}")
        blocks.append("")

    prompt = "\n".join(blocks)

    instructions = """
Solve MULTIPLE Year 5–8 maths questions from the same page.

Return ONLY raw JSON (no markdown) as an array.
Each item:
{
  "page_number": <integer or null>,
  "question_number": "<exact QID>",
  "correct_answer": "<final answer only, concise>"
}
""".strip()

    last_raw = ""
    for _ in range(attempts):
        resp = client.responses.create(
            model=TEXT_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=900,
        )
        raw = resp.output_text.strip()
        last_raw = raw

        try:
            arr = _parse_json_loose(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(arr, list):
            continue

        solved_by_key: Dict[Tuple[Optional[int], str], Dict[str, Any]] = {}
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            p = _norm_page(obj.get("page_number", page))
            qn = str(obj.get("question_number", "")).strip()
            if not qn:
                continue
            solved_by_key[_key(p, qn)] = {
                "page_number": p,
                "question_number": qn,
                "correct_answer": str(obj.get("correct_answer", "")).strip(),
            }

        out: List[Dict[str, Any]] = []
        for it in items:
            p = _norm_page(it.get("page_number"))
            qn = str(it.get("question_number", "")).strip()
            k = _key(p, qn)
            if k not in solved_by_key:
                return None
            out.append(solved_by_key[k])

        return out

    print("Solve parse failed after retries (batch). Raw output below:\n")
    print(last_raw)
    return None


def _judge_equivalence_llm(
    client: OpenAI,
    page: Optional[int],
    qn: str,
    qtext: str,
    student_answer: str,
    correct_answer: str,
    attempts: int = RETRIES,
) -> Dict[str, Any]:
    prompt = "\n".join(
        [
            f"PAGE: {page if page is not None else 'null'}",
            f"QID: {qn}",
            f"Question: {qtext}",
            f"Student final answer: {student_answer}",
            f"Correct answer: {correct_answer}",
        ]
    )

    instructions = """
Decide if the student's final answer is correct.

Rules:
- score = 1 if correct (equivalent forms allowed)
- score = 0 if wrong, BLANK, or UNREADABLE
- max_score is always 1

Return ONLY raw JSON (no markdown):
{
  "page_number": <integer or null>,
  "question_number": "<exact QID>",
  "score": 0 or 1,
  "max_score": 1,
  "feedback": "<short>"
}
""".strip()

    last_raw = ""
    for _ in range(attempts):
        resp = client.responses.create(
            model=TEXT_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=350,
        )
        raw = resp.output_text.strip()
        last_raw = raw

        try:
            obj = _parse_json_loose(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        return {
            "page_number": _norm_page(obj.get("page_number", page)),
            "question_number": str(obj.get("question_number", qn)).strip() or qn,
            "score": int(obj.get("score", 0)),
            "max_score": 1,
            "feedback": str(obj.get("feedback", "")).strip(),
        }

    print("Judge parse failed after retries. Raw output below:\n")
    print(last_raw)

    return {
        "page_number": page,
        "question_number": qn,
        "score": 0,
        "max_score": 1,
        "feedback": "Judging failed (unparseable model output). Needs manual review.",
    }


def grade_math_qa(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not qa_items:
        return []

    client = _client()

    # group by page
    grouped: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
    for qa in qa_items:
        grouped[_norm_page(qa.get("page_number"))].append(qa)

    # STEP 1: SOLVE (no student answers involved)
    solved_map: Dict[Tuple[Optional[int], str], str] = {}

    for page in sorted(grouped.keys(), key=lambda x: (-1 if x is None else x)):
        items = grouped[page]

        solved_batch: Optional[List[Dict[str, Any]]] = None
        if _should_batch_page(items):
            solved_batch = _solve_page_batch(client, page, items, attempts=RETRIES)

        if solved_batch is not None:
            for s in solved_batch:
                solved_map[_key(s.get("page_number"), s.get("question_number", ""))] = str(
                    s.get("correct_answer", "")
                ).strip()
        else:
            for it in items:
                s = _solve_one(client, it, attempts=RETRIES)
                solved_map[_key(s.get("page_number"), s.get("question_number", ""))] = str(
                    s.get("correct_answer", "")
                ).strip()

    #  STEP 2: SCORE (code first, LLM only if needed)
    out: List[Dict[str, Any]] = []

    for qa in qa_items:
        page = _norm_page(qa.get("page_number"))
        qn = str(qa.get("question_number", "")).strip()
        qtext = str(qa.get("question_text", "")).strip()
        student = _clean_text(str(qa.get("student_answer", "") or "").strip())
        correct = _clean_text(solved_map.get(_key(page, qn), "") or "")

        if student.upper() in BLANK_TOKENS:
            out.append(
                {
                    "page_number": page,
                    "question_number": qn,
                    "score": 0,
                    "max_score": 1,
                    "correct_answer": correct,
                    "feedback": "No answer provided." if student.upper() in {"", "BLANK"} else "Answer unreadable.",
                }
            )
            continue

        # deterministic numeric check (single numbers OR numeric sequences)
        num_eq = _numbers_equivalent(student, correct)
        if num_eq is True:
            out.append(
                {
                    "page_number": page,
                    "question_number": qn,
                    "score": 1,
                    "max_score": 1,
                    "correct_answer": correct,
                    "feedback": "Correct.",
                }
            )
            continue
        if num_eq is False:
            out.append(
                {
                    "page_number": page,
                    "question_number": qn,
                    "score": 0,
                    "max_score": 1,
                    "correct_answer": correct,
                    "feedback": f"Incorrect. Correct answer: {correct}" if correct else "Incorrect.",
                }
            )
            continue

        # fallback: text/logic answers (or unclear formats)
        judged = _judge_equivalence_llm(
            client,
            page=page,
            qn=qn,
            qtext=qtext,
            student_answer=student,
            correct_answer=correct,
            attempts=RETRIES,
        )
        out.append(
            {
                "page_number": page,
                "question_number": qn,
                "score": int(judged.get("score", 0)),
                "max_score": 1,
                "correct_answer": correct,
                "feedback": str(judged.get("feedback", "")).strip(),
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
            "page_number": 1,
            "question_number": "2",
            "question_text": "What is 2/5 as a decimal?",
            "student_answer": "0.40",
            "student_working": "",
        },
    ]
    res = grade_math_qa(fake_qa)
    print(json.dumps(res, indent=2, ensure_ascii=False))
