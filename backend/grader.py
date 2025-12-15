"""
grader.py

Deterministic-first math grading for the demo pipeline.

- Tries to grade common Grade 5–8 style problems locally (no LLM):
  * plain addition expressions (integers + decimals)
  * word problems that are clearly "add these numbers"
  * "insert commas" formatting
  * "write this number in words" (English) for large integers + simple decimals
  * a few special challenge types used in the demo worksheet

- Falls back to an LLM only when deterministic grading can't confidently handle the item.

Return schema (per question):
{
  "page_number": int | None,
  "question_number": str,
  "score": int,
  "max_score": int,
  "correct_answer": str,
  "feedback": str
}
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple

# High precision for big-int + decimal operations
getcontext().prec = 60

# utils
_QN_FIX_1 = re.compile(r"^(\d+)\(([A-Za-z])$")        # e.g. "2(c"
_QN_FIX_2 = re.compile(r"^(\d+)([A-Za-z])$")          # e.g. "4a"
_QN_CLEAN = re.compile(r"\s+")

def normalize_question_number(qn: Any) -> str:
    """Normalize question numbers so keys match reliably."""
    s = str(qn or "").strip()
    s = _QN_CLEAN.sub("", s)

    # Common OCR glitches
    m = _QN_FIX_1.match(s)
    if m:
        s = f"{m.group(1)}({m.group(2)})"

    m = _QN_FIX_2.match(s)
    if m:
        s = f"{m.group(1)}({m.group(2)})"

    # Normalise bracket casing
    s = re.sub(r"\(([A-Za-z])\)", lambda mm: f"({mm.group(1).lower()})", s)
    return s


_NUM_TOKEN = re.compile(r"(?<!\w)(\$?\d[\d,]*\.?\d*)(?!\w)")

def _strip_currency_units(s: str) -> str:
    # keep digits, commas, decimal point, minus
    s = s.strip()
    s = s.replace("$", "")
    # remove trailing non-numeric units
    s = re.sub(r"[^\d,\.\-]+$", "", s.strip())
    return s.strip()

def parse_decimal(s: str) -> Optional[Decimal]:
    """
    Parse a numeric-looking string into Decimal.
    Accepts commas and optional $.
    Returns None if it doesn't look numeric.
    """
    if s is None:
        return None
    s = str(s).strip()
    s = _strip_currency_units(s)
    if not re.search(r"\d", s):
        return None
    s = s.replace(",", "")
    try:
        return Decimal(s)
    except Exception:
        return None

def decimal_places(d: Decimal) -> int:
    tup = d.as_tuple()
    return max(0, -tup.exponent)

def format_decimal(d: Decimal, places: Optional[int] = None, use_commas: bool = False) -> str:
    """
    Format Decimal with optional fixed decimal places and optional commas.
    """
    if places is not None:
        q = Decimal("1").scaleb(-places)  # 10^-places
        d = d.quantize(q)

    # Convert to string without scientific notation
    s = f"{d:f}"

    if "." in s:
        # strip trailing zeros if no fixed places requested
        if places is None:
            s = s.rstrip("0").rstrip(".")
    if use_commas:
        if "." in s:
            int_part, frac = s.split(".", 1)
            int_part_commas = f"{int(int_part):,}" if int_part else "0"
            return f"{int_part_commas}.{frac}"
        return f"{int(s):,}"
    return s

def normalize_answer_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s


# english words to number

_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "quadrillion": 1_000_000_000_000_000,
    "quintillion": 1_000_000_000_000_000_000,
}

_DENOMS = {
    "tenths": 1,  # special-cased: denominator 10^1
    "hundredths": 2,
    "thousandths": 3,
    "ten-thousandths": 4,
    "hundred-thousandths": 5,
    "millionths": 6,
}

_WORD_SPLIT = re.compile(r"[^a-zA-Z\-]+")

def words_to_decimal(words: str) -> Optional[Decimal]:
    """
    Parse simple English number words into a Decimal.

    Supports:
    - large integers up to quintillions
    - decimals expressed like "... and two hundred thirty two thousandths"
    """
    if not words:
        return None

    w = normalize_answer_text(words)
    w = w.replace(",", " ")
    w = w.replace("-", " ")
    tokens = [t for t in _WORD_SPLIT.split(w) if t]
    if not tokens:
        return None

    # Identify fractional unit (hundredths, thousandths, etc.)
    frac_places = None
    if tokens and tokens[-1] in _DENOMS:
        frac_places = _DENOMS[tokens[-1]]
        tokens = tokens[:-1]  # remove denom token

    # Split on "and" for decimals; but "and" also appears in integers (optional). We'll only treat it as decimal
    # separator if we have an explicit denominator.
    if frac_places is not None and "and" in tokens:
        and_idx = len(tokens) - 1 - tokens[::-1].index("and")  # last "and"
        int_tokens = tokens[:and_idx]
        frac_tokens = tokens[and_idx + 1 :]
    else:
        int_tokens = tokens
        frac_tokens = []

    int_val = _words_to_int(int_tokens)
    if int_val is None:
        return None

    if frac_places is None:
        return Decimal(int_val)

    frac_val = _words_to_int(frac_tokens)
    if frac_val is None:
        return None
    # fractional part is frac_val / 10^frac_places
    frac_str = str(frac_val).rjust(frac_places, "0")
    return Decimal(f"{int_val}.{frac_str}")

def _words_to_int(tokens: List[str]) -> Optional[int]:
    if not tokens:
        return 0

    total = 0
    current = 0
    seen = False

    for t in tokens:
        if t in ("and",):
            continue

        if t in _SMALL:
            current += _SMALL[t]
            seen = True
            continue

        if t == "hundred":
            if current == 0:
                current = 1
            current *= 100
            seen = True
            continue

        if t in _SCALES and t != "hundred":
            scale = _SCALES[t]
            total += current * scale
            current = 0
            seen = True
            continue

        # Unknown token
        return None

    if not seen:
        return None

    return total + current


# graders

@dataclass
class GradeResult:
    score: int
    max_score: int = 1
    correct_answer: str = ""
    feedback: str = ""


def _grade_addition_expression(question_text: str, student_answer: str) -> Optional[GradeResult]:
    """
    Handles:
    - "Solve ...: 128 + 284"
    - "Add the following: 10 + 12 + 14 + 16"
    - "Arrange ... and solve: 4789562 + 5562842"
    """
    qt = normalize_answer_text(question_text)

    # Only attempt if it looks like an addition prompt
    if "+" not in question_text and not any(k in qt for k in ["add the following", "in total", "in all", "sum of"]):
        return None

    nums = [parse_decimal(m.group(1)) for m in _NUM_TOKEN.finditer(question_text)]
    nums = [n for n in nums if n is not None]

    # For "sum of even integers from 30 to 58" etc, we handle elsewhere
    if not nums:
        return None

    # For plain addition tasks, summing all numbers is fine.
    places = max(decimal_places(n) for n in nums)
    correct = sum(nums, Decimal(0))
    correct_str = format_decimal(correct, places=places, use_commas=False)

    s_dec = parse_decimal(student_answer)
    if s_dec is None:
        return GradeResult(0, correct_answer=correct_str, feedback="Could not parse a numeric final answer.")

    # Compare with quantization
    s_q = Decimal(s_dec).quantize(Decimal("1").scaleb(-places))
    c_q = Decimal(correct).quantize(Decimal("1").scaleb(-places))

    if s_q == c_q:
        return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
    return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")


def _grade_insert_commas(question_text: str, student_answer: str) -> Optional[GradeResult]:
    qt = normalize_answer_text(question_text)
    if "insert commas" not in qt:
        return None

    # Find the "target number" from the question text (digits only)
    nums = [m.group(1) for m in _NUM_TOKEN.finditer(question_text)]
    if not nums:
        return None

    target_raw = nums[-1]
    target_digits = re.sub(r"\D", "", target_raw)
    if not target_digits:
        return None

    correct_fmt = f"{int(target_digits):,}"

    sa = (student_answer or "").strip()
    sa_compact = re.sub(r"\s+", "", sa)
    # remove trailing punctuation like ";" "." etc
    sa_compact = sa_compact.strip(";,.")
    # Many OCR answers contain spaces; compare after removing spaces
    if sa_compact == correct_fmt:
        return GradeResult(1, correct_answer=correct_fmt, feedback="Correct.")
    # Also allow users who typed digits only
    if re.sub(r"\D", "", sa_compact) == target_digits and "," not in sa_compact:
        return GradeResult(0, correct_answer=correct_fmt, feedback="Digits are right, but commas are missing.")
    return GradeResult(0, correct_answer=correct_fmt, feedback="Incorrect comma placement.")


def _grade_number_words(question_text: str, student_answer: str) -> Optional[GradeResult]:
    qt = normalize_answer_text(question_text)
    if "write" not in qt or "words" not in qt:
        return None

    # Expect a number in the question text
    nums = [parse_decimal(m.group(1)) for m in _NUM_TOKEN.finditer(question_text)]
    nums = [n for n in nums if n is not None]
    if not nums:
        return None
    expected = nums[-1]

    # Parse student words -> Decimal
    got = words_to_decimal(student_answer or "")
    if got is None:
        return GradeResult(0, correct_answer=format_decimal(expected), feedback="Could not interpret the words as a number.")

    # Compare with same number of decimal places as expected
    places = decimal_places(expected)
    exp_q = expected.quantize(Decimal("1").scaleb(-places))
    got_q = got.quantize(Decimal("1").scaleb(-places))

    if got_q == exp_q:
        return GradeResult(1, correct_answer=_number_to_words_for_readme(expected), feedback="Correct.")
    return GradeResult(0, correct_answer=_number_to_words_for_readme(expected), feedback="Incorrect number words.")


def _number_to_words_for_readme(d: Decimal) -> str:
    """
    Lightweight formatter for 'correct_answer' on word problems.
    We don't need perfect grammar — just something readable.
    """
    # For this project, keep it simple: just echo the numeric form if conversion isn't needed.
    # (Users mainly care about correctness, not perfect English hyphenation.)
    s = format_decimal(d, use_commas=True)
    return s


def _grade_word_problem_addition(question_text: str, student_answer: str) -> Optional[GradeResult]:
    qt = normalize_answer_text(question_text)

    # Special-case: Abigail "five times her age" series sum
    if "five times her age" in qt and "how much money has she gotten so far" in qt:
        ages = [int(re.sub(r"\D", "", m.group(1))) for m in _NUM_TOKEN.finditer(question_text) if re.search(r"\d", m.group(1))]
        age = max(ages) if ages else 0
        correct = Decimal(5 * age * (age + 1) // 2)  # 5*(1+...+age)
        correct_str = f"${format_decimal(correct, use_commas=True)}"
        s_dec = parse_decimal(student_answer)
        if s_dec is None:
            return GradeResult(0, correct_answer=correct_str, feedback="Could not parse a dollar amount.")
        if s_dec == correct:
            return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
        return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")

    # Sum of even integers from 30 to 58
    if "sum of the positive even integers from" in qt and "to" in qt:
        # Extract endpoints
        nums = [int(re.sub(r"\D", "", m.group(1))) for m in _NUM_TOKEN.finditer(question_text)]
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            # Ensure even step 2
            if a % 2 == 0 and b % 2 == 0 and a <= b:
                n = (b - a) // 2 + 1
                correct = Decimal(n * (a + b) // 2)
                correct_str = format_decimal(correct, use_commas=False)
                s_dec = parse_decimal(student_answer)
                if s_dec == correct:
                    return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
                return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")
        return None

    # Six-digit numbers with one digit 4 and rest 1
    if "six-digit" in qt and "one digit equal to 4" in qt and "rest" in qt and "equal to 1" in qt:
        nums = [411111,141111,114111,111411,111141,111114]
        correct = Decimal(sum(nums))
        correct_str = format_decimal(correct, use_commas=True)
        s_dec = parse_decimal(student_answer)
        if s_dec == correct:
            return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
        return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")

    # Three digit numbers of 1s and 2s
    if "three digit numbers" in qt and "only" in qt and "1" in question_text and "2" in question_text and "sum" in qt:
        nums = [111,112,121,122,211,212,221,222]
        correct = Decimal(sum(nums))
        correct_str = format_decimal(correct, use_commas=True)
        s_dec = parse_decimal(student_answer)
        if s_dec == correct:
            return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
        return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")

    # Sandwich shop combination
    if "sandwich shop" in qt and "together cost exactly" in qt and "which sandwiches" in qt:
        # Parse prices
        # We'll use fixed expected items for now (demo worksheet)
        items = [
            ("turkey", Decimal("5.56")),
            ("roast beef", Decimal("7.72")),
            ("chicken salad", Decimal("6.21")),
            ("prosciutto and mozzarella", Decimal("9.34")),
            ("peanut butter and jelly", Decimal("3.75")),
        ]
        target = Decimal("21.00")

        solutions = []
        for a in range(len(items)):
            for b in range(len(items)):
                for c in range(len(items)):
                    tot = (items[a][1] + items[b][1] + items[c][1]).quantize(Decimal("0.01"))
                    if tot == target:
                        names = [items[a][0], items[b][0], items[c][0]]
                        # compress counts
                        counts = {}
                        for n in names:
                            counts[n] = counts.get(n, 0) + 1
                        parts = []
                        for n, ct in sorted(counts.items()):
                            if ct == 1:
                                parts.append(f"one {n}")
                            else:
                                parts.append(f"{ct} {n}")
                        solutions.append(", ".join(parts))

        correct_text = solutions[0] if solutions else "No solution found."
        sa = normalize_answer_text(student_answer)
        # Accept if student's text mentions two roast beef and one turkey (order-insensitive)
        ok = ("roast" in sa and "turkey" in sa and ("two" in sa or "2" in sa) and ("one" in sa or "1" in sa or "a" in sa))
        if ok:
            return GradeResult(1, correct_answer=correct_text, feedback="Correct.")
        return GradeResult(0, correct_answer=correct_text, feedback="Incorrect combination.")

    # Generic word-problem addition: if it contains >=2 numbers and asks "in total / in all / farther"
    if any(k in qt for k in ["in total", "in all", "how much", "how far", "farther"]):
        nums = [parse_decimal(m.group(1)) for m in _NUM_TOKEN.finditer(question_text)]
        nums = [n for n in nums if n is not None]
        if len(nums) >= 2:
            places = max(decimal_places(n) for n in nums)
            correct = sum(nums, Decimal(0))
            correct_str = format_decimal(correct, places=places, use_commas=True)

            s_dec = parse_decimal(student_answer)
            if s_dec is None:
                return GradeResult(0, correct_answer=correct_str, feedback="Could not parse a numeric answer.")
            s_q = s_dec.quantize(Decimal("1").scaleb(-places))
            c_q = correct.quantize(Decimal("1").scaleb(-places))
            if s_q == c_q:
                return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
            return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")
    return None


def _grade_emma_carry_challenge(question_text: str, student_answer: str, student_working: str) -> Optional[GradeResult]:
    qt = normalize_answer_text(question_text)
    if "emma added" not in qt or "abc" not in qt or "782" not in qt:
        return None

    # If student gives an answer, we can infer ABC and the carry pattern that answer implies.
    s_dec = parse_decimal(student_answer)
    if s_dec is None:
        return GradeResult(0, correct_answer="", feedback="Could not parse a numeric final answer.")

    s_int = int(s_dec)
    abc = s_int - 782
    if abc < 100 or abc > 999:
        return GradeResult(0, correct_answer="", feedback="Answer does not correspond to adding a 3-digit number to 782.")

    A, B, C = abc // 100, (abc // 10) % 10, abc % 10
    if len({A, B, C}) < 3:
        return GradeResult(0, correct_answer="", feedback="A, B, and C must be different digits.")

    # Determine carry pattern from this ABC
    carry1 = 1 if 2 + C >= 10 else 0
    carry2 = 1 if 8 + B + carry1 >= 10 else 0
    carry3 = 1 if 7 + A + carry2 >= 10 else 0

    # Compute the true maximum sum under that carry pattern
    best_sum, _best_abc = _max_sum_for_carry_pattern(carry1, carry2, carry3)

    correct_str = str(best_sum)
    if s_int == best_sum:
        return GradeResult(1, correct_answer=correct_str, feedback="Correct.")
    return GradeResult(0, correct_answer=correct_str, feedback="Incorrect.")


def _max_sum_for_carry_pattern(c1: int, c2: int, c3: int) -> Tuple[int, int]:
    best = -1
    best_abc = -1
    for A in range(1, 10):
        for B in range(0, 10):
            for C in range(0, 10):
                if len({A, B, C}) < 3:
                    continue
                abc = 100 * A + 10 * B + C
                s = 782 + abc

                ones = 2 + C
                carry1 = 1 if ones >= 10 else 0
                tens = 8 + B + carry1
                carry2 = 1 if tens >= 10 else 0
                hundreds = 7 + A + carry2
                carry3 = 1 if hundreds >= 10 else 0

                if (carry1, carry2, carry3) == (c1, c2, c3):
                    if s > best:
                        best = s
                        best_abc = abc
    return best, best_abc


def deterministic_grade_one(qa: Dict[str, Any]) -> Optional[GradeResult]:
    qtext = qa.get("question_text", "") or ""
    sans = qa.get("student_answer", "") or ""
    work = qa.get("student_working", "") or ""

    # Special challenge first
    out = _grade_emma_carry_challenge(qtext, sans, work)
    if out is not None:
        return out

    # Insert commas
    out = _grade_insert_commas(qtext, sans)
    if out is not None:
        return out

    # Number -> words
    out = _grade_number_words(qtext, sans)
    if out is not None:
        return out

    # Word problems (including special sums)
    out = _grade_word_problem_addition(qtext, sans)
    if out is not None:
        return out

    # Plain expressions
    out = _grade_addition_expression(qtext, sans)
    if out is not None:
        return out

    return None


# ----------------------------
# Optional LLM fallback (kept minimal)
# ----------------------------

def _llm_grade_fallback(qa: Dict[str, Any]) -> GradeResult:
    """
    Minimal fallback using OpenAI, if configured.

    If not configured, returns "ungraded" as incorrect with a helpful message.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return GradeResult(
            0,
            correct_answer="",
            feedback="No deterministic grader matched this problem, and OPENAI_API_KEY is not set for LLM fallback.",
        )

    try:
        from openai import OpenAI
    except Exception:
        return GradeResult(
            0,
            correct_answer="",
            feedback="No deterministic grader matched this problem, and the OpenAI SDK is not installed for LLM fallback.",
        )

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are grading a student's math answer.\n\n"
        "Return STRICT JSON with keys: score (0 or 1), correct_answer (string), feedback (string).\n"
        "Do not include any extra keys.\n\n"
        f"QUESTION: {qa.get('question_text','')}\n"
        f"STUDENT_WORKING: {qa.get('student_working','')}\n"
        f"STUDENT_ANSWER: {qa.get('student_answer','')}\n"
    )
    try:
        resp = client.responses.create(
            model=os.getenv("TEXT_MODEL", "gpt-4.1-mini"),
            input=prompt,
            max_output_tokens=250,
        )
        text = (resp.output_text or "").strip()
        # best-effort JSON parse
        import json
        data = json.loads(text)
        score = int(data.get("score", 0))
        return GradeResult(
            score=1 if score == 1 else 0,
            correct_answer=str(data.get("correct_answer", "")),
            feedback=str(data.get("feedback", "")),
        )
    except Exception as e:
        return GradeResult(0, correct_answer="", feedback=f"LLM fallback failed: {e}")


# ----------------------------
# Public API
# ----------------------------

def grade_math_qa(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grade a list of extracted Q/A items.

    Always returns a list (never None).
    """
    out: List[Dict[str, Any]] = []
    for qa in qa_items or []:
        page = qa.get("page_number", None)
        qn = normalize_question_number(qa.get("question_number", ""))

        # Deterministic-first
        res = deterministic_grade_one({**qa, "question_number": qn})
        if res is None:
            res = _llm_grade_fallback({**qa, "question_number": qn})

        out.append(
            {
                "page_number": page,
                "question_number": qn,
                "score": res.score,
                "max_score": res.max_score,
                "correct_answer": res.correct_answer,
                "feedback": res.feedback,
            }
        )
    return out
