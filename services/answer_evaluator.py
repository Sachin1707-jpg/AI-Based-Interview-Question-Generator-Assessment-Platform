# ============================================================
# services/answer_evaluator.py
# P_112 — AI-Based Interview Question Generator
# Evaluates a candidate's answer using OpenAI and returns a
# structured result: score, feedback, correct/missing concepts,
# improvement suggestions.
# ============================================================

import os
import json
import time
import logging
from typing import Any

from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
load_dotenv()

_MODEL          = "gpt-4o"
_FALLBACK_MODEL = "gpt-4o-mini"
_TIMEOUT_SEC    = 45
_MAX_RETRIES    = 2
_RETRY_DELAY    = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# EVALUATION RESULT SCHEMA
# ─────────────────────────────────────────────────────────────
# The service always returns a dict with exactly these keys:
#
# {
#   "score":                int (0–10),
#   "percentage":           float (0–100),
#   "grade":                str ("Excellent" | "Good" | "Average" | "Poor"),
#   "feedback":             str  — overall qualitative summary,
#   "correct_concepts":     list[str] — things the candidate got right,
#   "missing_concepts":     list[str] — important things omitted,
#   "improvement_suggestions": list[str] — concrete next steps,
#   "rubric_breakdown":     list[{"criterion": str, "points_awarded": int,
#                                 "points_possible": int, "comment": str}]
# }


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def _build_eval_prompt(
    question: str,
    expected_answer: str,
    rubric: list[dict],
    candidate_answer: str,
    maximum_score: int = 10,
) -> tuple[str, str]:
    """Build the system and user prompt for answer evaluation."""

    rubric_text = "\n".join(
        f"  - {r.get('criterion', 'Criterion')} ({r.get('points', 1)} pts)"
        for r in (rubric if isinstance(rubric, list) else [])
    )

    system_prompt = (
        "You are a strict but fair senior technical interviewer and assessor. "
        "Your role is to evaluate a candidate's interview answer against an "
        "expected answer and a scoring rubric. "
        "You MUST respond with a single valid JSON object and nothing else — "
        "no markdown, no code fences, no preamble, no trailing text."
    )

    user_prompt = f"""Evaluate the following interview answer:

QUESTION
--------
{question}

EXPECTED ANSWER (Model Answer)
-------------------------------
{expected_answer}

EVALUATION RUBRIC (total = {maximum_score} points)
---------------------------------------------------
{rubric_text}

CANDIDATE'S ANSWER
------------------
{candidate_answer if candidate_answer.strip() else "[No answer provided]"}

EVALUATION INSTRUCTIONS
-----------------------
1. Score each rubric criterion individually based on how well the candidate addressed it.
2. Sum the criterion scores to get the total score (0–{maximum_score}).
3. Be strict but fair — partial credit is acceptable for partially correct answers.
4. An empty or irrelevant answer should score 0.
5. Identify specific concepts the candidate demonstrated correctly.
6. Identify important concepts that were missing or incorrect.
7. Provide 2–4 concrete, actionable improvement suggestions.
8. Keep feedback concise but informative (2–4 sentences).

REQUIRED OUTPUT (return exactly this JSON structure, nothing else):
{{
  "score": <integer 0 to {maximum_score}>,
  "percentage": <float 0.0 to 100.0>,
  "grade": "<Excellent | Good | Average | Poor>",
  "feedback": "<2-4 sentence overall assessment>",
  "correct_concepts": ["<concept the candidate got right>", ...],
  "missing_concepts": ["<important concept that was absent or wrong>", ...],
  "improvement_suggestions": ["<actionable suggestion>", ...],
  "rubric_breakdown": [
    {{
      "criterion": "<criterion text>",
      "points_awarded": <int>,
      "points_possible": <int>,
      "comment": "<one-line justification>"
    }}
  ]
}}"""

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────
# GRADE HELPER
# ─────────────────────────────────────────────────────────────
def _derive_grade(score: int, maximum_score: int) -> str:
    pct = (score / maximum_score * 100) if maximum_score > 0 else 0
    if pct >= 80:
        return "Excellent"
    elif pct >= 60:
        return "Good"
    elif pct >= 40:
        return "Average"
    return "Poor"


# ─────────────────────────────────────────────────────────────
# SCHEMA VALIDATOR & REPAIRER
# ─────────────────────────────────────────────────────────────
def _repair_evaluation(raw: dict, maximum_score: int = 10) -> dict:
    """
    Normalise and repair a raw evaluation dict from the LLM so it
    always has all required keys with correct types.
    """
    # score — clamp to [0, maximum_score]
    try:
        score = int(round(float(raw.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(score, maximum_score))

    # percentage — derive from score if missing or wrong
    try:
        pct = float(raw.get("percentage", 0))
    except (TypeError, ValueError):
        pct = 0.0
    if not (0.0 <= pct <= 100.0):
        pct = round(score / maximum_score * 100, 1) if maximum_score else 0.0

    # grade
    grade = str(raw.get("grade", "")).strip()
    if grade not in {"Excellent", "Good", "Average", "Poor"}:
        grade = _derive_grade(score, maximum_score)

    # feedback
    feedback = str(raw.get("feedback", "")).strip()
    if not feedback:
        feedback = "The answer was evaluated but no detailed feedback was generated."

    # list fields
    def _safe_list(key: str) -> list[str]:
        val = raw.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(x).strip() for x in val if str(x).strip()]

    correct_concepts  = _safe_list("correct_concepts")
    missing_concepts  = _safe_list("missing_concepts")
    improvements      = _safe_list("improvement_suggestions")

    # rubric_breakdown
    raw_rb = raw.get("rubric_breakdown", [])
    rubric_breakdown = []
    if isinstance(raw_rb, list):
        for item in raw_rb:
            if not isinstance(item, dict):
                continue
            rubric_breakdown.append({
                "criterion":      str(item.get("criterion", "Criterion")),
                "points_awarded": max(0, int(item.get("points_awarded", 0))),
                "points_possible": max(1, int(item.get("points_possible", 1))),
                "comment":        str(item.get("comment", "")),
            })

    return {
        "score":                   score,
        "percentage":              pct,
        "grade":                   grade,
        "feedback":                feedback,
        "correct_concepts":        correct_concepts,
        "missing_concepts":        missing_concepts,
        "improvement_suggestions": improvements,
        "rubric_breakdown":        rubric_breakdown,
    }


def _validate_evaluation(ev: dict, maximum_score: int = 10) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors = []
    score = ev.get("score")
    if not isinstance(score, int) or not (0 <= score <= maximum_score):
        errors.append(f"score must be an int in [0, {maximum_score}], got {score!r}")
    if ev.get("grade") not in {"Excellent", "Good", "Average", "Poor"}:
        errors.append(f"grade invalid: {ev.get('grade')!r}")
    if not ev.get("feedback"):
        errors.append("feedback is empty")
    return errors


# ─────────────────────────────────────────────────────────────
# JSON EXTRACTOR
# ─────────────────────────────────────────────────────────────
def _extract_json_object(raw: str) -> str:
    """
    Extract the JSON object from raw text even if wrapped in
    markdown fences or preceded by explanatory text.
    """
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("`"):
        lines = raw.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "`":
            inner = inner[:-1]
        raw = "\n".join(inner).strip()

    # Find outermost { ... }
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


# ─────────────────────────────────────────────────────────────
# API CALLER
# ─────────────────────────────────────────────────────────────
def _call_openai(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call OpenAI with retry + fallback model. Returns raw text."""
    client = OpenAI(api_key=api_key, timeout=_TIMEOUT_SEC)
    models = [_MODEL, _FALLBACK_MODEL]
    last_error: Exception | None = None

    for model in models:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("Evaluator API call: model=%s attempt=%d", model, attempt)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,   # low temperature → consistent scoring
                    max_tokens=1024,
                    response_format={"type": "json_object"} if model == "gpt-4o" else None,
                )
                return resp.choices[0].message.content or ""

            except APITimeoutError as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

            except APIConnectionError as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

            except APIStatusError as e:
                last_error = e
                if e.status_code in {401, 403, 429}:
                    break
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)

    raise RuntimeError(
        f"Evaluation API failed after all retries. Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def evaluate_answer(
    question:        str,
    expected_answer: str,
    rubric:          list[dict],
    candidate_answer: str,
    maximum_score:   int = 10,
) -> dict[str, Any]:
    """
    Evaluate a candidate's answer against the expected answer and rubric.

    Parameters
    ----------
    question         : The interview question text
    expected_answer  : The model/expected answer
    rubric           : List of {"criterion": str, "points": int}
    candidate_answer : The candidate's typed response
    maximum_score    : Maximum possible score (default 10)

    Returns
    -------
    dict with keys:
        score, percentage, grade, feedback,
        correct_concepts, missing_concepts,
        improvement_suggestions, rubric_breakdown

    Raises
    ------
    EnvironmentError  : API key not set
    RuntimeError      : API call failed after all retries
    ValueError        : Response could not be parsed or is critically invalid
    """
    # ── Guard: API key ─────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )

    # ── Handle empty answer immediately ───────────────────────
    if not candidate_answer.strip():
        return {
            "score": 0,
            "percentage": 0.0,
            "grade": "Poor",
            "feedback": "No answer was provided. A score of 0 has been assigned.",
            "correct_concepts": [],
            "missing_concepts": ["No concepts demonstrated — answer was empty."],
            "improvement_suggestions": [
                "Always attempt an answer, even if unsure.",
                "Review the topic thoroughly before the next attempt.",
            ],
            "rubric_breakdown": [
                {
                    "criterion": r.get("criterion", "Criterion"),
                    "points_awarded": 0,
                    "points_possible": r.get("points", 1),
                    "comment": "Not addressed.",
                }
                for r in (rubric if isinstance(rubric, list) else [])
            ],
        }

    # ── Build prompt ───────────────────────────────────────────
    system_prompt, user_prompt = _build_eval_prompt(
        question, expected_answer, rubric, candidate_answer, maximum_score
    )

    # ── Call API ───────────────────────────────────────────────
    raw_text = _call_openai(system_prompt, user_prompt, api_key)

    # ── Extract JSON ───────────────────────────────────────────
    json_str = _extract_json_object(raw_text)

    # ── Parse JSON ─────────────────────────────────────────────
    try:
        raw_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse evaluation response as JSON.\n"
            f"Error: {e}\nSnippet: {json_str[:200]}"
        ) from e

    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected a JSON object, got: {type(raw_data).__name__}")

    # ── Repair & normalise ─────────────────────────────────────
    evaluation = _repair_evaluation(raw_data, maximum_score)

    # ── Validate (non-critical — log but don't raise) ──────────
    val_errors = _validate_evaluation(evaluation, maximum_score)
    if val_errors:
        logger.warning("Evaluation validation warnings: %s", val_errors)

    logger.info(
        "Answer evaluated: score=%d/%d grade=%s",
        evaluation["score"], maximum_score, evaluation["grade"],
    )

    return evaluation
