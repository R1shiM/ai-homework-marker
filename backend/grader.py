import json
import os
import re
import string
from decimal import Decimal, InvalidOperation, getcontext
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Increase precision for big-number arithmetic (money, large integers, etc.)
getcontext().prec = 50

TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-5-chat-latest")

RETRIES = 2  # for LLM-only (non-deterministic) questions


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


def _strip_code_fences(s: str) -> str:
    # Just in case, but json_schema should prevent fences.
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _normalize_text_for_compare(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace("×", "*")
    s = re.sub(r"\s+", " ", s)
    # remove surrounding punctuation
    s = s.strip(string.punctuation + " ")
    return s


_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def _extract_first_number(s: str) -> Optional[str]:
    if not s:
        return None
    m = _NUM_RE.search(s)
    return m.group(0) if m else None


def _to_decimal(num_str: str) -> Optional[Decimal]:
    if not num_str:
        return None
    try:
        cleaned = num_str.replace(",", "").strip()
        # strip leading currency symbols
        cleaned = cleaned.replace("$", "")
        return Decimal(cleaned)
    except (InvalidOperation, AttributeError):
        return None


def _format_decimal(d: Decimal) -> str:
    # format without scientific notation, remove trailing zeros
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _numeric_equal(student: str, correct: str, tol: Decimal = Decimal("0")) -> bool:
    # tol=0 means exact numeric equality for Decimals
    sn = _extract_first_number(student)
    cn = _extract_first_number(correct)
    sd = _to_decimal(sn) if sn else None
    cd = _to_decimal(cn) if cn else None
    if sd is None or cd is None:
        return False
    return abs(sd - cd) <= tol


# -----------------------------
# Multi-part expansion
# -----------------------------
# Matches lines like:
#   128 + 284 = 412
#   4510044 - 3625931 = 884,113
_ARITH_LINE = re.compile(
    r"^\s*([0-9][0-9,\.]*)\s*([+\-*/])\s*([0-9][0-9,\.]*)\s*=\s*([0-9][0-9,\.]*)\s*$"
)

# Also handle "a + b" without "=" (sometimes in printed text)
_ARITH_EXPR = re.compile(r"^\s*([0-9][0-9,\.]*)\s*([+\-*/])\s*([0-9][0-9,\.]*)\s*$")


def _split_student_answer_list(ans: str) -> List[str]:
    """
    Turns things like:
      "523, 144, 139, 1555, 1929, 3679"
      "52 154 26664 215 422 166665"
    into a list of tokens (best-effort).
    """
    if not ans:
        return []
    # normalize separators
    s = ans.strip()
    s = s.replace("\n", " ").replace(";", " ").replace(" and ", " ")
    # split on commas OR whitespace
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p.strip()]
    return parts


def _suffix_letters(n: int) -> str:
    # 0->a, 1->b, ..., 25->z, 26->aa...
    out = ""
    n += 1
    while n > 0:
        n -= 1
        out = chr(ord("a") + (n % 26)) + out
        n //= 26
    return out


def _expand_multi_part(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If we see multiple arithmetic lines in working, split into sub-questions:
      qn "2" -> "2a", "2b", ...
    If student_answer is a list, align it by order.
    """
    page = _norm_page(qa.get("page_number"))
    qn = str(qa.get("question_number", "")).strip()
    qtext = str(qa.get("question_text", "")).strip()
    sans = str(qa.get("student_answer", "")).strip()
    work = str(qa.get("student_working", "")).strip()

    if not qn:
        return [qa]

    lines = [ln.strip() for ln in work.splitlines() if ln.strip()]
    arith_lines = []
    for ln in lines:
        m = _ARITH_LINE.match(ln.replace("×", "*"))
        if m:
            a, op, b, c = m.groups()
            arith_lines.append((a, op, b, c, ln))

    # If working contains 2+ arithmetic lines, treat as multi-part
    if len(arith_lines) >= 2:
        ans_parts = _split_student_answer_list(sans)
        out: List[Dict[str, Any]] = []
        for i, (a, op, b, c, ln) in enumerate(arith_lines):
            sub_qn = f"{qn}{_suffix_letters(i)}"
            # prefer explicit "= c" from working; else align from student_answer list
            sub_student = c
            if ans_parts and i < len(ans_parts):
                sub_student = ans_parts[i]

            out.append(
                {
                    "page_number": page,
                    "question_number": sub_qn,
                    "question_text": f"{qtext}  [{a} {op} {b}]",
                    "student_answer": sub_student,
                    "student_working": ln,
                    "_derived_from": qn,
                }
            )
        return out

    return [qa]


def _expand_all(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for qa in qa_items:
        expanded.extend(_expand_multi_part(qa))
    return expanded

def _try_grade_arithmetic(qa: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    If the question is a simple binary arithmetic expression (from our expansion),
    compute correct answer in Python and grade deterministically.
    """
    page = _norm_page(qa.get("page_number"))
    qn = str(qa.get("question_number", "")).strip()
    qtext = str(qa.get("question_text", "")).strip()
    sans = str(qa.get("student_answer", "")).strip()
    work = str(qa.get("student_working", "")).strip()

    # Prefer extracting from the bracketed "[a op b]" we add during expansion.
    bracket = re.search(r"\[([0-9][0-9,\.]*)\s*([+\-*/])\s*([0-9][0-9,\.]*)\]", qtext)
    expr = None
    if bracket:
        expr = f"{bracket.group(1)} {bracket.group(2)} {bracket.group(3)}"
    else:
        # fallback: if student_working is literally "a op b = c"
        m = _ARITH_LINE.match(work.replace("×", "*"))
        if m:
            expr = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        else:
            # last fallback: if question_text itself is "a op b"
            m2 = _ARITH_EXPR.match(qtext.replace("×", "*"))
            if m2:
                expr = f"{m2.group(1)} {m2.group(2)} {m2.group(3)}"

    if not expr:
        return None

    m = _ARITH_EXPR.match(expr.replace("×", "*").strip())
    if not m:
        return None

    a_s, op, b_s = m.groups()
    a = _to_decimal(a_s)
    b = _to_decimal(b_s)
    if a is None or b is None:
        return None

    try:
        if op == "+":
            correct = a + b
        elif op == "-":
            correct = a - b
        elif op == "*":
            correct = a * b
        elif op == "/":
            # avoid division by zero
            if b == 0:
                return None
            correct = a / b
        else:
            return None
    except Exception:
        return None

    correct_str = _format_decimal(correct)

    # If student answer is blank/unreadable
    if sans.strip().upper() in {"", "BLANK", "UNREADABLE"}:
        score = 0
        feedback = "No answer provided."
    else:
        # numeric compare (tolerance 0 by default)
        ok = _numeric_equal(sans, correct_str, tol=Decimal("0"))
        score = 1 if ok else 0
        feedback = "Correct." if ok else f"Incorrect. Correct answer: {correct_str}"

    return {
        "page_number": page,
        "question_number": qn,
        "score": score,
        "max_score": 1,
        "correct_answer": correct_str,
        "feedback": feedback,
    }

def _llm_grade_one(client: OpenAI, qa: Dict[str, Any], attempts: int = RETRIES) -> Dict[str, Any]:
    page = _norm_page(qa.get("page_number"))
    qn = str(qa.get("question_number", "")).strip()
    qtext = str(qa.get("question_text", "")).strip()
    sans = str(qa.get("student_answer", "")).strip()
    work = str(qa.get("student_working", "")).strip()

    prompt_lines = []
    prompt_lines.append(f"PAGE: {page if page is not None else 'null'}")
    prompt_lines.append(f"QUESTION_NUMBER: {qn}")
    prompt_lines.append(f"QUESTION_TEXT: {qtext}")
    if work:
        prompt_lines.append(f"STUDENT_WORKING: {work}")
    prompt_lines.append(f"STUDENT_FINAL_ANSWER: {sans}")

    qa_block = "\n".join(prompt_lines)

    instructions = """
You are marking ONE Year 5–8 maths question.

Rules:
- score = 1 if the student's final answer is mathematically correct (equivalent forms allowed)
- score = 0 if wrong, BLANK, or UNREADABLE
- max_score is always 1

Return STRICT JSON matching the schema.
""".strip()

    schema = {
        "name": "grade_item",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_number": {"type": ["integer", "null"]},
                "question_number": {"type": "string"},
                "score": {"type": "integer", "enum": [0, 1]},
                "max_score": {"type": "integer", "enum": [1]},
                "correct_answer": {"type": "string"},
                "feedback": {"type": "string"},
            },
            "required": ["page_number", "question_number", "score", "max_score", "correct_answer", "feedback"],
        },
        "strict": True,
    }

    last_raw = ""
    for _ in range(attempts):
        resp = client.responses.create(
            model=TEXT_MODEL,
            instructions=instructions,
            input=qa_block,
            temperature=0,
            #response_format={"type": "json_schema", "json_schema": schema},
            max_output_tokens=350,
        )

        raw = (resp.output_text or "").strip()
        last_raw = raw

        try:
            obj = json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        # enforce page/qn fallback
        obj["page_number"] = _norm_page(obj.get("page_number", page))
        obj["question_number"] = str(obj.get("question_number", qn)).strip() or qn
        return obj

    # fallback
    print("LLM grading failed (invalid JSON). Raw output:\n")
    print(last_raw)
    return {
        "page_number": page,
        "question_number": qn,
        "score": 0,
        "max_score": 1,
        "correct_answer": "",
        "feedback": "Grading failed (invalid model output). Needs manual review.",
    }

def grade_math_qa(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not qa_items:
        return []

    client = _client()

    # 1) Expand multi-part arithmetic into subquestions (2a, 2b, ...)
    expanded = _expand_all(qa_items)

    out: List[Dict[str, Any]] = []
    seen_keys = set()

    for qa in expanded:
        page = _norm_page(qa.get("page_number"))
        qn = str(qa.get("question_number", "")).strip()
        if not qn:
            continue

        k = _key(page, qn)
        if k in seen_keys:
            # avoid collisions if OCR duplicates question numbers on same page
            # append a suffix to force uniqueness
            i = 1
            new_qn = f"{qn}_{i}"
            while _key(page, new_qn) in seen_keys:
                i += 1
                new_qn = f"{qn}_{i}"
            qa["question_number"] = new_qn
            qn = new_qn
            k = _key(page, qn)

        seen_keys.add(k)

        # 2) Deterministic grade for arithmetic
        det = _try_grade_arithmetic(qa)
        if det is not None:
            out.append(det)
            continue

        # 3) Otherwise LLM grade with strict schema
        out.append(_llm_grade_one(client, qa, attempts=RETRIES))

    return out


if __name__ == "__main__":
    fake_qa = [
        {
            "page_number": 1,
            "question_number": "2",
            "question_text": "Solve the following problems.",
            "student_answer": "",
            "student_working": "128 + 284 = 412\n34 + 28 = 62\n64 + 48 = 112",
        },
        {
            "page_number": 1,
            "question_number": "3",
            "question_text": "What is the smallest prime bigger than 17?",
            "student_answer": "19",
            "student_working": "",
        },
    ]
    res = grade_math_qa(fake_qa)
    print(json.dumps(res, indent=2, ensure_ascii=False))
