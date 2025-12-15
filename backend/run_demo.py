import json
import os
from typing import List, Dict, Any, Tuple, Optional

from ocr_pipeline import extract_qa_from_file
from grader import grade_math_qa, normalize_question_number


def _key(item: Dict[str, Any]) -> Tuple[Optional[int], str]:
    page = item.get("page_number")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    qn = normalize_question_number(item.get("question_number", ""))
    return (page, qn)


def _pretty_summary(
    qa_items: List[Dict[str, Any]],
    graded_items: List[Dict[str, Any]],
) -> None:
    graded_items = graded_items or []
    graded_by_key = {_key(g): g for g in graded_items}

    total = 0
    total_max = 0

    print("\n=== Grading Summary ===\n")

    for qa in qa_items:
        k = _key(qa)
        g = graded_by_key.get(k, {})

        qn = normalize_question_number(qa.get("question_number", ""))
        qtext = qa.get("question_text", "")
        sans = qa.get("student_answer", "")
        work = qa.get("student_working", "")
        page = qa.get("page_number")

        score = g.get("score", 0)
        max_score = g.get("max_score", 1)
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

    # Normalise question_number immediately so grading + printing use the same key
    for qa in qa_items:
        qa["question_number"] = normalize_question_number(qa.get("question_number", ""))

    print("Raw Q/A JSON:\n")
    print(json.dumps(qa_items, indent=2, ensure_ascii=False))

    print("\n[2/2] Grading...\n")
    graded_items = grade_math_qa(qa_items)
    print("Raw grading JSON:\n")
    print(json.dumps(graded_items, indent=2, ensure_ascii=False))

    _pretty_summary(qa_items, graded_items)


if __name__ == "__main__":
    main()
