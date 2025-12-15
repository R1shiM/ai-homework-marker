import json
import os
import re
import string
from typing import List, Dict, Any, Tuple, Optional

from ocr_pipeline import extract_qa_from_file
from grader import grade_math_qa


def _key(item: Dict[str, Any]) -> Tuple[Optional[int], str]:
    page = item.get("page_number")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    qn = str(item.get("question_number", "")).strip()
    return (page, qn)


_ARITH_LINE = re.compile(
    r"^\s*([0-9][0-9,\.]*)\s*([+\-*/])\s*([0-9][0-9,\.]*)\s*=\s*([0-9][0-9,\.]*)\s*$"
)


def _split_student_answer_list(ans: str) -> List[str]:
    """
    Turns things like:
      "523, 144, 139, 1555, 1929, 3679"
      "52 154 26664 215 422 166665"
    into a list of tokens (best-effort).
    """
    if not ans:
        return []
    s = ans.strip().replace("\n", " ").replace(";", " ")
    # split on commas OR whitespace
    parts = re.split(r"[,\s]+", s)
    return [p for p in parts if p.strip()]


def _suffix_letters(n: int) -> str:
    # 0->a, 1->b, ..., 25->z, 26->aa...
    out = ""
    n += 1
    while n > 0:
        n -= 1
        out = chr(ord("a") + (n % 26)) + out
        n //= 26
    return out


def _expand_for_summary(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mirror the grader's "split multi-line arithmetic working into subquestions".
    This makes the summary keys match graded outputs like 2a, 2b, ...
    """
    expanded: List[Dict[str, Any]] = []

    for qa in qa_items:
        page = qa.get("page_number")
        qn = str(qa.get("question_number", "")).strip()
        qtext = str(qa.get("question_text", "")).strip()
        sans = str(qa.get("student_answer", "")).strip()
        work = str(qa.get("student_working", "")).strip()

        # Only expand clean numeric question numbers (avoid weird cases like "2(a)")
        if not re.fullmatch(r"\d+", qn):
            expanded.append(qa)
            continue

        lines = [ln.strip() for ln in work.splitlines() if ln.strip()]
        arith_lines = []
        for ln in lines:
            m = _ARITH_LINE.match(ln.replace("×", "*"))
            if m:
                a, op, b, c = m.groups()
                arith_lines.append((a, op, b, c, ln))

        # Expand only when we truly have multiple distinct arithmetic lines
        if len(arith_lines) >= 2:
            ans_parts = _split_student_answer_list(sans)

            for i, (a, op, b, c, ln) in enumerate(arith_lines):
                sub_qn = f"{qn}{_suffix_letters(i)}"

                # Prefer explicit RHS from working, but allow aligned list from student_answer if present
                sub_student = c
                if ans_parts and i < len(ans_parts):
                    sub_student = ans_parts[i]

                expanded.append(
                    {
                        "page_number": page,
                        "question_number": sub_qn,
                        "question_text": f"{qtext}  [{a} {op} {b}]",
                        "student_answer": sub_student,
                        "student_working": ln,
                    }
                )
        else:
            expanded.append(qa)

    return expanded


def _pretty_summary(
    qa_items_for_summary: List[Dict[str, Any]],
    graded_items: List[Dict[str, Any]],
) -> None:
    graded_by_key = {_key(g): g for g in graded_items}

    total = 0
    total_max = 0

    print("\n=== Grading Summary ===\n")

    for qa in qa_items_for_summary:
        k = _key(qa)
        g = graded_by_key.get(k, {})

        qn = qa.get("question_number", "")
        qtext = qa.get("question_text", "")
        sans = qa.get("student_answer", "")
        work = qa.get("student_working", "")
        page = qa.get("page_number")

        score = int(g.get("score", 0) or 0)
        max_score = int(g.get("max_score", 1) or 1)
        correct_answer = g.get("correct_answer", "")
        feedback = g.get("feedback", "")

        total += score
        total_max += max_score

        if page is not None:
            print(f"Question {qn} (page {page})")
        else:
            print(f"Question {qn}")

        print(f"  Text: {qtext}")
        if work:
            print(f"  Student working: {work}")
        print(f"  Student final answer: {sans}")
        print(f"  -> Score: {score}/{max_score}")
        if correct_answer:
            print(f"  -> Correct answer: {correct_answer}")
        if feedback:
            print(f"  -> Feedback: {feedback}")
        print()

    print("=== Overall ===")
    if total_max > 0:
        pct = 100 * total / total_max
        print(f"Total: {total}/{total_max} ({pct:.1f}%)")
    else:
        print("No questions detected / graded")


def main() -> None:
    input_path = os.path.join("..", "sample_data", "sample_input.pdf")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Drop a worksheet PDF/image there or change the path."
        )

    print(f"Using input file: {input_path}")

    print("\n[1/2] OCR: extracting questions + answers...\n")
    qa_items = extract_qa_from_file(input_path)
    print("Raw Q/A JSON:\n")
    print(json.dumps(qa_items, indent=2, ensure_ascii=False))

    # Expand for reporting so keys match grader output (2a, 2b, ...)
    qa_items_for_summary = _expand_for_summary(qa_items)

    print("\n[2/2] Grading...\n")
    graded_items = grade_math_qa(qa_items)
    print("Raw grading JSON:\n")
    print(json.dumps(graded_items, indent=2, ensure_ascii=False))

    _pretty_summary(qa_items_for_summary, graded_items)


if __name__ == "__main__":
    main()
