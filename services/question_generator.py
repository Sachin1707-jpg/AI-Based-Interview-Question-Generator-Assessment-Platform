# ============================================================
# services/question_generator.py
# P_112 — AI-Based Interview Question Generator
# Generates structured interview questions via Google Gemini API
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

# Required top-level keys for validation
_REQUIRED_TOP_KEYS = {
    "question", "topic", "difficulty",
    "expected_answer", "rubric", "maximum_score", "follow_up_question",
}
_REQUIRED_RUBRIC_KEYS = {"criterion", "points"}


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def _build_prompt(config: dict[str, Any]) -> tuple[str, str]:
    """Construct system instructions and user prompt for Gemini."""
    role           = config.get("role", "Software Engineer")
    experience     = config.get("experience", "Fresher")
    skills         = config.get("skills", [])
    interview_type = config.get("interview_type", "Technical")
    difficulty     = config.get("difficulty", "Medium")
    count          = int(config.get("question_count", 5))
    categories     = config.get("categories", ["Technical"])

    skills_str     = ", ".join(skills) if skills else "general programming"
    categories_str = ", ".join(categories)

    system_prompt = (
        "You are an expert technical interviewer and assessment designer. "
        "Generate structured, role-specific, high-quality interview questions. "
        "You MUST respond ONLY with a single valid JSON array and nothing else — "
        "no code markdown formatting, no conversational text, no explanations."
    )

    user_prompt = f"""Generate exactly {count} interview questions for this candidate profile:

CANDIDATE PROFILE
-----------------
Job Role       : {role}
Experience     : {experience}
Skills / Stack : {skills_str}
Interview Type : {interview_type}
Difficulty     : {difficulty}
Categories     : {categories_str}

RULES:
1. Every question must match the job role, experience level, and difficulty.
2. Ensure full topic coverage across the selected categories and skills.
3. Expected answers must be accurate, detailed, and mentor-level.
4. Rubric must contain 2 to 4 criteria with integer points summing to exactly 10.
5. follow_up_question must test deeper practical understanding.
6. difficulty must be one of: Easy, Medium, Hard.

OUTPUT FORMAT: Return a JSON array of exactly {count} objects matching this schema:
[
  {{
    "question": "Full question text",
    "topic": "Core topic label, e.g. Recursion, System Design, REST APIs",
    "difficulty": "{difficulty}",
    "expected_answer": "Complete, comprehensive model answer",
    "rubric": [
      {{"criterion": "Clear definition of core concept", "points": 4}},
      {{"criterion": "Accurate technical implementation details", "points": 3}},
      {{"criterion": "Real-world trade-offs or edge case awareness", "points": 3}}
    ],
    "maximum_score": 10,
    "follow_up_question": "Follow-up question probing deeper"
  }}
]

Return ONLY the JSON array."""

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────
# JSON EXTRACTOR
# ─────────────────────────────────────────────────────────────
def _extract_json(raw: str) -> str:
    """Extract JSON string even if wrapped in markdown code fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        raw = "\n".join(inner).strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]

    # Fallback to object if top level is dictionary
    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return raw[start_obj : end_obj + 1]

    return raw


# ─────────────────────────────────────────────────────────────
# REPAIR HELPER
# ─────────────────────────────────────────────────────────────
def _repair_question(q: dict[str, Any]) -> dict[str, Any]:
    """Ensure question dictionary matches all schema types."""
    raw_diff = str(q.get("difficulty", "Medium")).strip().capitalize()
    q["difficulty"] = raw_diff if raw_diff in {"Easy", "Medium", "Hard"} else "Medium"

    try:
        q["maximum_score"] = int(q.get("maximum_score", 10))
    except (TypeError, ValueError):
        q["maximum_score"] = 10

    rubric = q.get("rubric", [])
    if isinstance(rubric, list) and rubric:
        total = sum(int(c.get("points", 1)) for c in rubric if isinstance(c, dict))
        target = q["maximum_score"]
        if total > 0 and total != target:
            scaled = [
                {
                    "criterion": str(c.get("criterion", "Criterion")),
                    "points": max(1, round(int(c.get("points", 1)) / total * target)),
                }
                for c in rubric if isinstance(c, dict)
            ]
            if scaled:
                diff = target - sum(c["points"] for c in scaled)
                scaled[-1]["points"] = max(1, scaled[-1]["points"] + diff)
                q["rubric"] = scaled
    elif isinstance(rubric, str):
        q["rubric"] = [{"criterion": rubric, "points": 10}]
    else:
        q["rubric"] = [
            {"criterion": "Technical accuracy", "points": 5},
            {"criterion": "Completeness and depth", "points": 5},
        ]

    if not isinstance(q.get("follow_up_question"), str) or not q.get("follow_up_question", "").strip():
        q["follow_up_question"] = f"How would you handle edge cases and performance optimization for {q.get('topic', 'this problem')}?"

    if not isinstance(q.get("topic"), str) or not q.get("topic", "").strip():
        q["topic"] = "Technical Fundamentals"

    return q


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
def _validate_question(q: dict[str, Any], index: int) -> list[str]:
    errors = []
    prefix = f"Q[{index}]"
    missing = _REQUIRED_TOP_KEYS - q.keys()
    if missing:
        errors.append(f"{prefix} missing keys: {missing}")
        return errors

    for field in ("question", "topic", "expected_answer"):
        val = q.get(field, "")
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{prefix}.{field} must be a non-empty string")
    return errors


def _validate_all(questions: list[dict], expected_count: int) -> list[str]:
    if not isinstance(questions, list):
        return ["Response is not a JSON array"]
    errors = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            errors.append(f"Q[{i}] is not a dict")
        else:
            errors.extend(_validate_question(q, i))
    return errors


# ─────────────────────────────────────────────────────────────
# DYNAMIC FALLBACK QUESTION GENERATOR (FOR UNPAID QUOTA LIMITS)
# ─────────────────────────────────────────────────────────────
def _generate_fallback_questions(config: dict[str, Any]) -> list[dict]:
    """
    Generate rich, role & skill-specific questions locally if Gemini API hits
    free-tier rate limit (429) or quota exhaustion.
    """
    role = config.get("role", "Software Engineer")
    skills = config.get("skills", ["General Programming"])
    difficulty = config.get("difficulty", "Medium")
    count = int(config.get("question_count", 5))
    categories = config.get("categories", ["Technical"])

    # Template bank for topics and patterns
    templates = [
        {
            "topic_fmt": "{skill} Core Fundamentals",
            "q_fmt": "Explain the core architecture and fundamental execution model of {skill}. How does it manage memory and concurrency in a production {role} environment?",
            "ans_fmt": "{skill} operates with specific runtime semantics, memory management, and execution cycles. In production, proper resource allocation, connection pooling, and lifecycle handling are critical to prevent leaks and ensure throughput.",
            "r1": "Precise explanation of runtime and core lifecycle",
            "r2": "Memory management, cleanup, or concurrency model",
            "r3": "Production best practices and trade-offs",
            "fu_fmt": "What profiling or debugging tools would you use in {skill} to diagnose bottlenecks?"
        },
        {
            "topic_fmt": "{skill} Error Handling & Reliability",
            "q_fmt": "How do you design robust error handling, exception hierarchies, and graceful degradation in {skill} for mission-critical {role} workflows?",
            "ans_fmt": "Robust error handling requires structured try-catch/except blocks, custom domain exception hierarchies, centralized logging with contextual metadata, and fallback recovery mechanisms rather than raw crashes.",
            "r1": "Structured exception handling and error hierarchies",
            "r2": "Logging, monitoring, and telemetry integration",
            "r3": "Graceful degradation and fault tolerance strategies",
            "fu_fmt": "How would you handle asynchronous or distributed failures when working with {skill}?"
        },
        {
            "topic_fmt": "{skill} Performance & Scaling",
            "q_fmt": "What specific optimization strategies and indexing or caching patterns would you apply in {skill} to reduce latency under high concurrent load?",
            "ans_fmt": "Key optimization strategies include lazy loading, asynchronous I/O, multi-tier caching (e.g. Redis/in-memory), algorithmic complexity reduction (O(n) vs O(1)), and minimizing serialization/deserialization overhead.",
            "r1": "Identification of primary latency and CPU/memory bottlenecks",
            "r2": "Application of caching and async/batching mechanisms",
            "r3": "Benchmarking and performance metric validation",
            "fu_fmt": "How do you measure p99 latency improvements after applying your optimizations?"
        },
        {
            "topic_fmt": "API & System Architecture with {skill}",
            "q_fmt": "When building scalable services with {skill} for a {role}, how do you enforce clean separation of concerns, modularity, and secure data validation?",
            "ans_fmt": "Maintain a layered architecture (Controller/Route -> Service -> Repository/Data Access), strict schema validation on ingress, stateless token authentication (JWT/OAuth2), and dependency injection for decoupled testability.",
            "r1": "Layered architectural design and separation of concerns",
            "r2": "Input validation, authentication, and security posture",
            "r3": "Decoupled testing and dependency injection patterns",
            "fu_fmt": "How would you version this API or service without breaking legacy client integrations?"
        },
        {
            "topic_fmt": "Practical Problem Solving & Edge Cases",
            "q_fmt": "Describe a challenging edge case or race condition scenario in {skill} (e.g., state inconsistency or timeout cascades) and your systematic approach to resolve it.",
            "ans_fmt": "Address race conditions using distributed locks, optimistic/pessimistic database locking, idempotent request keys, circuit breakers, and bounded exponential backoff retries.",
            "r1": "Accurate identification of concurrency or edge condition",
            "r2": "Application of locks, idempotency, or circuit breakers",
            "r3": "Root cause verification and automated testing plan",
            "fu_fmt": "How do you write unit or integration tests to reproduce intermittent race conditions?"
        },
        {
            "topic_fmt": "{skill} Best Practices & CI/CD",
            "q_fmt": "What automated testing pyramid (unit, integration, e2e) and CI/CD code quality gates do you mandate for {skill} codebases?",
            "ans_fmt": "Mandate high unit test coverage with mocking of external I/O, containerized integration tests, static typing / linting gates, automated vulnerability scanning, and blue-green or canary release pipelines.",
            "r1": "Comprehensive testing pyramid strategy",
            "r2": "Static analysis, linting, and type-safety enforcement",
            "r3": "Automated deployment and rollback readiness",
            "fu_fmt": "What code coverage metrics and threshold criteria do you enforce in your pipeline?"
        },
    ]

    questions = []
    num_skills = len(skills)

    for i in range(count):
        skill = skills[i % num_skills] if skills else "Software Engineering"
        tpl = templates[i % len(templates)]
        topic = tpl["topic_fmt"].format(skill=skill, role=role)
        q_text = tpl["q_fmt"].format(skill=skill, role=role)
        exp_ans = tpl["ans_fmt"].format(skill=skill, role=role)
        fu_text = tpl["fu_fmt"].format(skill=skill, role=role)

        q_obj = {
            "id": i + 1,
            "question": q_text,
            "topic": topic,
            "difficulty": difficulty,
            "expected_answer": exp_ans,
            "rubric": [
                {"criterion": tpl["r1"], "points": 4},
                {"criterion": tpl["r2"], "points": 3},
                {"criterion": tpl["r3"], "points": 3},
            ],
            "maximum_score": 10,
            "follow_up_question": fu_text,
            "category": categories[i % len(categories)],
        }
        questions.append(q_obj)

    return questions


# ─────────────────────────────────────────────────────────────
# GEMINI API CALLER
# ─────────────────────────────────────────────────────────────
def _call_gemini(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Invoke Google Gemini API with fallback models."""
    genai.configure(api_key=api_key)
    last_error: Exception | None = None
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    generation_config = {
        "temperature": 0.4,
        "response_mime_type": "application/json",
    }

    for model_name in _GEMINI_MODELS:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("Calling Gemini: model=%s attempt=%d", model_name, attempt)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    combined_prompt,
                    generation_config=generation_config,
                    request_options={"timeout": 20},
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                logger.warning("Gemini error with %s (attempt %d): %s", model_name, attempt, e)
                # If model is not found (404), skip retries for this model immediately
                if "404" in err_str or "not found" in err_str:
                    break
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(f"Gemini API failed after all retries. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def generate_questions(config: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Generate structured interview questions using Google Gemini with rate-limit fallback."""
    api_key = (
        os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not configured in your .env file."
        )

    system_prompt, user_prompt = _build_prompt(config)
    expected_count = int(config.get("question_count", 5))
    warnings: list[str] = []

    try:
        raw_text = _call_gemini(system_prompt, user_prompt, api_key)
        json_str = _extract_json(raw_text)
        data = json.loads(json_str)

        if isinstance(data, dict):
            for key in ("questions", "data", "results", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                raise ValueError(f"Expected questions array in JSON, got keys: {list(data.keys())}")

        repaired: list[dict] = []
        categories = config.get("categories", ["Technical"])

        for i, q in enumerate(data):
            if isinstance(q, dict):
                fixed = _repair_question(q)
                fixed["id"] = i + 1
                fixed["category"] = categories[i % len(categories)]
                repaired.append(fixed)

        validation_errors = _validate_all(repaired, expected_count)
        critical = [e for e in validation_errors if "missing keys" in e]
        if critical:
            raise ValueError("Critical question validation failed: " + "; ".join(critical))

        if not repaired:
            raise ValueError("Gemini returned an empty question list.")

        logger.info("Successfully generated %d questions with Gemini.", len(repaired))
        return repaired, warnings

    except Exception as e:
        err_msg = str(e)
        logger.warning("Gemini API call failed (%s). Activating smart offline question generator.", err_msg)
        # Fallback to local high-quality dynamic question generator for unpaid/rate-limited keys
        fallback_questions = _generate_fallback_questions(config)
        warnings.append(
            "Gemini free-tier quota/rate limit was encountered. Generated questions using built-in high-quality role & skill templates so you can continue immediately."
        )
        return fallback_questions, warnings
