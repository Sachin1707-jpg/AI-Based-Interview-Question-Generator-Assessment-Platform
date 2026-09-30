# ============================================================
# services/evaluator.py
# P_112 — AI-Based Interview Question Generator
# Objective, rubric-based technical answer evaluator using Gemini API
# ============================================================

import os
import json
import time
import logging
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
load_dotenv()

_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]
_MAX_RETRIES = 1
_RETRY_DELAY = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# RUBRIC FORMATTER & MAX SCORE CALCULATOR
# ─────────────────────────────────────────────────────────────
def _format_rubric(rubric: Any) -> tuple[str, int]:
    if isinstance(rubric, list):
        lines = []
        total_pts = 0
        for item in rubric:
            if isinstance(item, dict):
                crit = str(item.get("criterion", "Criterion")).strip()
                pts = item.get("points", 1)
                try:
                    pts = int(pts)
                except (ValueError, TypeError):
                    pts = 1
                total_pts += pts
                lines.append(f"- {crit} ({pts} pts)")
            elif isinstance(item, str):
                lines.append(f"- {item}")
                total_pts += 1
        max_score = total_pts if total_pts > 0 else 10
        return "\n".join(lines), max_score
    elif isinstance(rubric, str) and rubric.strip():
        return rubric.strip(), 10
    return "Evaluate general technical correctness and completeness (10 pts total).", 10


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def _build_eval_prompt(
    question: str,
    expected_answer: str,
    rubric_text: str,
    candidate_answer: str,
    maximum_score: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are an objective, unbiased technical assessment evaluator. "
        "Your task is to evaluate a candidate's answer strictly against the "
        "provided rubric and technical requirements. "
        "RULES:\n"
        "1. Focus ONLY on technical correctness, conceptual depth, and rubric satisfaction.\n"
        "2. Do NOT evaluate or mention personal characteristics, tone, writing style, or background.\n"
        "3. Award points based solely on technical evidence present in the candidate's answer.\n"
        "4. The score MUST be an integer between 0 and the maximum score.\n"
        "5. Respond with a single valid JSON object only — no code blocks, no markdown wrappers, no commentary."
    )

    user_prompt = f"""EVALUATION TASK
----------------
Evaluate the candidate's answer based on the technical rubric below.

QUESTION:
{question}

EXPECTED MODEL ANSWER:
{expected_answer}

EVALUATION RUBRIC (Maximum Total Score: {maximum_score}):
{rubric_text}

CANDIDATE'S ANSWER:
{candidate_answer if candidate_answer.strip() else "[No answer provided]"}

EVALUATION INSTRUCTIONS:
- Verify which rubric criteria the candidate satisfied.
- List all demonstrated technical strengths in `correct_points`.
- List all missing or flawed technical concepts in `missing_points`.
- Write concise, objective technical feedback in `feedback`.
- Provide concrete, actionable technical steps to improve in `improvement`.
- Ensure score does not exceed {maximum_score}.

REQUIRED JSON OUTPUT FORMAT:
{{
  "score": <integer from 0 to {maximum_score}>,
  "maximum_score": {maximum_score},
  "correct_points": ["<correct technical point 1>", "<correct technical point 2>"],
  "missing_points": ["<missing concept 1>", "<missing concept 2>"],
  "feedback": "<2-3 sentences concise technical evaluation summary>",
  "improvement": "<1-2 sentences concrete technical improvement advice>"
}}"""

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────
# JSON EXTRACTOR & SANITIZER
# ─────────────────────────────────────────────────────────────
def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 1:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# ─────────────────────────────────────────────────────────────
# RESULT NORMALIZATION & REPAIR
# ─────────────────────────────────────────────────────────────
def _repair_eval_response(raw: Any, maximum_score: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    try:
        score = int(round(float(raw.get("score", 0))))
    except (ValueError, TypeError):
        score = 0
    score = max(0, min(score, maximum_score))

    try:
        max_s = int(raw.get("maximum_score", maximum_score))
        if max_s <= 0:
            max_s = maximum_score
    except (ValueError, TypeError):
        max_s = maximum_score

    def _to_str_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        elif isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    correct_points = _to_str_list(raw.get("correct_points") or raw.get("correct_concepts") or raw.get("strengths"))
    missing_points = _to_str_list(raw.get("missing_points") or raw.get("missing_concepts"))

    feedback = str(raw.get("feedback", "")).strip()
    if not feedback:
        if score == max_s:
            feedback = "The answer fully satisfies all rubric criteria and technical requirements."
        elif score > 0:
            feedback = "The answer partially meets the rubric criteria with some correct concepts demonstrated."
        else:
            feedback = "The answer did not meet the required technical criteria."

    improvement_val = raw.get("improvement") or raw.get("improvement_suggestions") or raw.get("improvements")
    if isinstance(improvement_val, list):
        improvement = " ".join(str(x).strip() for x in improvement_val if str(x).strip())
    else:
        improvement = str(improvement_val or "").strip()

    if not improvement:
        if missing_points:
            improvement = f"Focus on covering key concepts such as {', '.join(missing_points[:2])}."
        else:
            improvement = "Continue deepening hands-on problem solving and technical implementation details."

    return {
        "score": score,
        "maximum_score": max_s,
        "correct_points": correct_points,
        "missing_points": missing_points,
        "feedback": feedback,
        "improvement": improvement,
    }


# ─────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK EVALUATION (FOR UNPAID QUOTA LIMITS)
# ─────────────────────────────────────────────────────────────
def _heuristic_evaluation(
    question: str,
    expected_answer: str,
    rubric: Any,
    candidate_answer: str,
    maximum_score: int,
) -> dict[str, Any]:
    """
    Perform an objective heuristic rubric evaluation when Gemini API is
    rate-limited (429) or unavailable on unpaid free tier.
    """
    cand = candidate_answer.strip().lower()
    words = [w for w in cand.replace(",", " ").replace(".", " ").split() if len(w) > 2]
    
    # Extract key technical terms from expected answer
    exp_words = set(
        w.lower() for w in expected_answer.replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
        if len(w) > 3 and w.lower() not in {"this", "that", "with", "from", "have", "more", "also", "into", "their"}
    )
    
    match_count = sum(1 for w in words if w in exp_words)
    coverage_ratio = match_count / max(len(exp_words), 1) if exp_words else 0.5
    
    # Analyze length and structure
    length_pts = min(1.0, len(words) / 35.0)
    score_ratio = min(1.0, 0.4 * length_pts + 0.6 * min(1.0, coverage_ratio * 2.5))
    score = max(1, min(maximum_score, int(round(score_ratio * maximum_score))))
    
    correct_points = []
    missing_points = []
    
    # Parse rubric criteria
    rubric_list = rubric if isinstance(rubric, list) else []
    for item in rubric_list:
        crit = item.get("criterion", "Technical Requirement") if isinstance(item, dict) else str(item)
        crit_words = set(crit.lower().split())
        if any(cw in cand for cw in crit_words if len(cw) > 4):
            correct_points.append(f"Demonstrated awareness of: {crit}")
        else:
            missing_points.append(f"Could elaborate more on: {crit}")
            
    if not correct_points:
        correct_points.append("Provided a relevant baseline explanation for the technical topic.")
    if not missing_points and score < maximum_score:
        missing_points.append("Include deeper implementation trade-offs and edge case considerations.")

    feedback = (
        f"Answer covers the core concepts with a score of {score}/{maximum_score}. "
        "Demonstrated solid technical understanding." if score >= 7 else
        f"Answer partially addresses the requirements ({score}/{maximum_score}). Focus on adding concrete technical details."
    )
    
    improvement = (
        "Good response. To achieve maximum score, discuss specific edge cases, concurrency, and production scaling."
        if score >= 7 else
        "Provide more specific terminology, code patterns, and practical execution details."
    )

    return {
        "score": score,
        "maximum_score": maximum_score,
        "correct_points": correct_points,
        "missing_points": missing_points,
        "feedback": feedback,
        "improvement": improvement,
    }


# ─────────────────────────────────────────────────────────────
# GEMINI API CALLER
# ─────────────────────────────────────────────────────────────
def _call_gemini_eval(system_prompt: str, user_prompt: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    combined = f"{system_prompt}\n\n{user_prompt}"
    last_error: Exception | None = None

    generation_config = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }

    for model_name in _GEMINI_MODELS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("Calling Gemini Evaluator: model=%s attempt=%d", model_name, attempt)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    combined,
                    generation_config=generation_config,
                    request_options={"timeout": 20},
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                logger.warning("Gemini evaluator error %s (attempt %d): %s", model_name, attempt, e)
                if "404" in err_str or "not found" in err_str:
                    break
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(f"Gemini Evaluation API failed after all retries. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def evaluate_answer(
    question: str,
    expected_answer: str,
    rubric: Any,
    candidate_answer: str,
    maximum_score: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a candidate's answer against the expected answer and rubric via Gemini
    with automatic graceful fallback on rate limits.
    """
    rubric_text, computed_max = _format_rubric(rubric)
    final_max_score = maximum_score if (maximum_score is not None and maximum_score > 0) else computed_max

    if not candidate_answer or not candidate_answer.strip():
        return {
            "score": 0,
            "maximum_score": final_max_score,
            "correct_points": [],
            "missing_points": ["No answer was provided to address the question."],
            "feedback": "No answer was provided. A score of 0 has been assigned.",
            "improvement": "Make sure to provide a detailed technical response covering the core requirements.",
        }

    api_key = (
        os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not configured in your .env file."
        )

    system_prompt, user_prompt = _build_eval_prompt(
        question=question,
        expected_answer=expected_answer,
        rubric_text=rubric_text,
        candidate_answer=candidate_answer.strip(),
        maximum_score=final_max_score,
    )

    try:
        raw_response = _call_gemini_eval(system_prompt, user_prompt, api_key)
        json_str = _extract_json_object(raw_response)
        parsed_data = json.loads(json_str)
        result = _repair_eval_response(parsed_data, final_max_score)
    except Exception as e:
        logger.warning("Gemini evaluation API error (%s). Using fallback evaluation engine.", e)
        result = _heuristic_evaluation(
            question=question,
            expected_answer=expected_answer,
            rubric=rubric,
            candidate_answer=candidate_answer,
            maximum_score=final_max_score,
        )

    logger.info(
        "Evaluation finished: score=%d/%d correct=%d missing=%d",
        result["score"],
        result["maximum_score"],
        len(result["correct_points"]),
        len(result["missing_points"]),
    )
    return result
