## Overview

MPV for **AI homework checker** for Year 5–8 maths worksheets.  
This extracts the questions + student answers from pdf or image into structured JSON, then grades each question and produces a scored summary.

## How to run
### Prerequisites
- Python **3.10+** (3.11+ recommended)
- An OpenAI API key

### then create the venv with dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### make a .env file with OPENAI_API_KEY in there

### then find your worksheet image or pdf you want to mark and put it in as sample_data/sample_input.pdf
### (if you want to change filename update file path in run_demo i should fix that when im not sleepy)

### then run the code
```
cd backend
python run_demo.py
```
---

## What's happening here boss?

### 1) Input
- Accepts a worksheet as a **PDF (multi-page)** or a **single image**.

### 2) OCR + Question/Answer Extraction (i.e. this is the Vision phase)
- If input is a **PDF**, each page is rendered to an image.
- gpt 4.1 reads each page and returns a JSON array of question/answer objects:
  - `question_number` (e.g. `"1"`, `"2(a)"`)
  - `question_text`
  - `student_answer` (or `"BLANK"` / `"UNREADABLE"`)
  - `student_working` (if visible)
  - `page_number` (for PDFs only i suppose)

### 3) Grading (Text)
- The extracted items are passed to a **text model** (which i think right now is gpt-5-chat-latest) that:
  1. Works out the correct answer
  2. Compares it to the student’s final answer
  3. Outputs grading JSON:
     - `score` (currently `0` or `1`)
     - `max_score` (currently always `1`)
     - `correct_answer`
     - `feedback`

- The grader may grade **page-level batches** (to reduce calls/cost) when its safe, otherwise it grades **question-by-question**.

### 4) Reporting
- The demo script prints:
  - Raw extracted Q/A JSON
  - Raw grading JSON
  - A human-readable per-question summary + overall score

---

## Models used

This project uses two model roles:

- **Vision model (OCR/extraction)**  
  Reads worksheet pages and converts them into structured JSON (questions, answers, working).

- **Text model (grading)**  
  Computes correct answers, compares with student responses, and generates scores + feedback.

> Model names are configurable in code (e.g. `VISION_MODEL`, `TEXT_MODEL`).

---

## Current limitations

- **LLM grading can be wrong sometimes**  
  The grading model may make arithmetic/logic mistakes on tricky or multi-step questions.

- **OCR/extraction quality varies**  
  Handwriting, low-quality scans, unusual layouts, or overlapping text can cause missing or incorrect extraction.

- **Multi-part / grouped questions are hard for now agrh**  
  Worksheets often contain blocks like “Solve the following problems” where multiple answers belong to one question number. The current schema is optimized for **one question → one score** and may under-handle subparts.

- **Open-ended responses are subjective**  
  For explanation-style questions, grading and feedback can vary between runs.

- **Strict JSON outputs can fail**  
  Models sometimes include code fences or extra text. The code uses retries, but edge cases may still require manual review.
---
## Output format

### Extracted Q/A item
```json
{
  "page_number": 1,
  "question_number": "3",
  "question_text": "...",
  "student_answer": "...",
  "student_working": "..."
}
