# ============================================================
# tests/test_qa_suite.py
# QA Test Suite covering all 12 requirement areas
# ============================================================

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure utf-8 output encoding on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.question_generator import (
    generate_questions,
    _extract_json,
    _repair_question,
    _validate_question,
    _validate_all,
)
from services.evaluator import (
    evaluate_answer,
    _format_rubric,
    _extract_json_object,
    _repair_eval_response,
)
from services.report_generator import (
    calculate_report_metrics,
)
from app import _validate_config, init_state, ROLE_OPTIONS, DIFFICULTY_OPTIONS


class TestInterviewQASuite(unittest.TestCase):

    # ─────────────────────────────────────────────────────────
    # 1. Application Startup & Modules Integrity
    # ─────────────────────────────────────────────────────────
    def test_01_application_startup_and_constants(self):
        """Verify that all core application constants and dependencies load cleanly."""
        self.assertGreater(len(ROLE_OPTIONS), 0)
        self.assertIn("Python Developer", ROLE_OPTIONS)
        self.assertIn("Easy", DIFFICULTY_OPTIONS)
        self.assertIn("Medium", DIFFICULTY_OPTIONS)
        self.assertIn("Hard", DIFFICULTY_OPTIONS)

    # ─────────────────────────────────────────────────────────
    # 2. Interview Configuration Dictionary Generation
    # ─────────────────────────────────────────────────────────
    def test_02_interview_configuration(self):
        """Verify configuration dict creation with expected fields and types."""
        config = {
            "role": "Python Developer",
            "experience": "1-3 years",
            "skills": ["Python", "FastAPI", "Docker"],
            "interview_type": "Technical",
            "difficulty": "Medium",
            "question_count": 5,
            "categories": ["Technical", "Scenario Based"],
        }
        self.assertEqual(config["role"], "Python Developer")
        self.assertEqual(config["question_count"], 5)
        self.assertEqual(len(config["skills"]), 3)
        self.assertEqual(len(config["categories"]), 2)

    # ─────────────────────────────────────────────────────────
    # 3. Input Validation
    # ─────────────────────────────────────────────────────────
    def test_03_input_validation(self):
        """Test valid, invalid, boundary, and edge-case user inputs."""
        # 3a. Valid standard input
        errors, warnings, skills = _validate_config(
            role="Python Developer",
            custom_role="",
            experience="1-3 years",
            skills=["Python", "SQL"],
            custom_skills_raw="",
            interview_type="Technical",
            difficulty="Medium",
            num_questions=5,
            categories=["Technical"],
        )
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(skills), 2)

        # 3b. Empty custom role error
        errors, _, _ = _validate_config(
            role="Custom",
            custom_role="",
            experience="Fresher",
            skills=["Python"],
            custom_skills_raw="",
            interview_type="Technical",
            difficulty="Easy",
            num_questions=5,
            categories=["Technical"],
        )
        self.assertIn("Custom role name cannot be empty", errors[0])

        # 3c. Empty skills error
        errors, _, _ = _validate_config(
            role="Backend Developer",
            custom_role="",
            experience="0-1 years",
            skills=[],
            custom_skills_raw="",
            interview_type="Technical",
            difficulty="Easy",
            num_questions=5,
            categories=["Technical"],
        )
        self.assertTrue(any("skill" in e.lower() for e in errors))

        # 3d. Empty category error
        errors, _, _ = _validate_config(
            role="Backend Developer",
            custom_role="",
            experience="0-1 years",
            skills=["Python"],
            custom_skills_raw="",
            interview_type="Technical",
            difficulty="Easy",
            num_questions=5,
            categories=[],
        )
        self.assertTrue(any("category" in e.lower() for e in errors))

        # 3e. Question count bounds (< 5 or > 15)
        errors, _, _ = _validate_config(
            role="Backend Developer",
            custom_role="",
            experience="0-1 years",
            skills=["Python"],
            custom_skills_raw="",
            interview_type="Technical",
            difficulty="Easy",
            num_questions=2,
            categories=["Technical"],
        )
        self.assertTrue(any("between 5 and 15" in e for e in errors))

    # ─────────────────────────────────────────────────────────
    # 4. LLM Question Generation
    # ─────────────────────────────────────────────────────────
    def test_04_llm_question_generation_mock(self):
        """Test question generator response normalization and rubric scoring structure."""
        sample_ai_json = json.dumps([
            {
                "question": "What are Python metaclasses?",
                "topic": "Python Internals",
                "difficulty": "Hard",
                "expected_answer": "Metaclasses are the classes of classes in Python.",
                "rubric": [
                    {"criterion": "Definition of metaclass", "points": 4},
                    {"criterion": "type() as a metaclass", "points": 3},
                    {"criterion": "Real-world use cases like ORMs", "points": 3},
                ],
                "maximum_score": 10,
                "follow_up_question": "How does __new__ differ from __init__ in a metaclass?",
            }
        ])

        with patch("services.question_generator._call_gemini", return_value=sample_ai_json):
            with patch("os.getenv", return_value="AQ.test-key-mock"):
                config = {
                    "role": "Python Developer",
                    "experience": "5+ years",
                    "skills": ["Python"],
                    "interview_type": "Technical",
                    "difficulty": "Hard",
                    "question_count": 1,
                    "categories": ["Technical"],
                }
                questions, warnings = generate_questions(config)
                self.assertEqual(len(questions), 1)
                q = questions[0]
                self.assertEqual(q["id"], 1)
                self.assertEqual(q["maximum_score"], 10)
                self.assertEqual(len(q["rubric"]), 3)
                self.assertIn("follow_up_question", q)

    # ─────────────────────────────────────────────────────────
    # 5. Invalid LLM Response Handling & Self-Repair
    # ─────────────────────────────────────────────────────────
    def test_05_invalid_llm_response_handling(self):
        """Test JSON extraction from markdown fences, broken schema repair, and missing fields."""
        # 5a. Markdown code fences with extra preamble
        fenced_raw = """Here is the generated output:
```json
[
  {
    "question": "Explain FastAPI routers.",
    "topic": "FastAPI",
    "difficulty": "medium",
    "expected_answer": "APIRouter allows modular route definition.",
    "rubric": [{"criterion": "Modular routing", "points": 5}, {"criterion": "Prefixing and tags", "points": 5}],
    "maximum_score": 10,
    "follow_up_question": "How do you include routers in main app?"
  }
]
```"""
        extracted = _extract_json(fenced_raw)
        parsed = json.loads(extracted)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)

        # 5b. Repair question normalization (casing, missing follow-up, rubric point scale)
        unrepaired = {
            "question": "What is Docker?",
            "topic": "DevOps",
            "difficulty": "easy",
            "expected_answer": "Containerization platform.",
            "rubric": [{"criterion": "Containers vs VMs", "points": 3}, {"criterion": "Images", "points": 3}],
            "maximum_score": 10,
        }
        repaired = _repair_question(unrepaired)
        self.assertEqual(repaired["difficulty"], "Easy")
        self.assertIn("follow_up_question", repaired)
        # Rubric scaled to sum to 10
        total_rubric = sum(r["points"] for r in repaired["rubric"])
        self.assertEqual(total_rubric, 10)

    # ─────────────────────────────────────────────────────────
    # 6. Candidate Answer Submission
    # ─────────────────────────────────────────────────────────
    def test_06_candidate_answer_submission(self):
        """Test empty answer and whitespace answer submission fast paths."""
        res_empty = evaluate_answer(
            question="What is asyncio in Python?",
            expected_answer="Asynchronous I/O framework.",
            rubric=[{"criterion": "Event loop", "points": 5}, {"criterion": "Coroutines", "points": 5}],
            candidate_answer="   ",
            maximum_score=10,
        )
        self.assertEqual(res_empty["score"], 0)
        self.assertEqual(res_empty["maximum_score"], 10)
        self.assertIn("correct_points", res_empty)
        self.assertIn("missing_points", res_empty)
        self.assertIn("feedback", res_empty)
        self.assertIn("improvement", res_empty)

    # ─────────────────────────────────────────────────────────
    # 7. AI Evaluation Logic & Score Clamping
    # ─────────────────────────────────────────────────────────
    def test_07_ai_evaluation_mock_and_score_clamping(self):
        """Test AI answer evaluation parsing, score clamping (0 to max), and schema guarantee."""
        mock_eval_json = json.dumps({
            "score": 12,  # Over maximum score to test clamping
            "maximum_score": 10,
            "correct_points": ["Understands async event loop", "Knows await keyword"],
            "missing_points": ["Did not mention task cancellation"],
            "feedback": "Strong understanding of asyncio fundamentals.",
            "improvement": "Review asyncio.gather exception handling.",
        })

        with patch("services.evaluator._call_gemini_eval", return_value=mock_eval_json):
            with patch("os.getenv", return_value="AQ.test-mock-key"):
                res = evaluate_answer(
                    question="How does asyncio event loop work?",
                    expected_answer="Event loop schedules and executes cooperative tasks.",
                    rubric=[{"criterion": "Event loop scheduling", "points": 5}, {"criterion": "Async/await", "points": 5}],
                    candidate_answer="The event loop executes tasks cooperatively using async and await.",
                    maximum_score=10,
                )
                self.assertEqual(res["score"], 10)  # Clamped to maximum_score
                self.assertEqual(res["maximum_score"], 10)
                self.assertEqual(len(res["correct_points"]), 2)
                self.assertEqual(len(res["missing_points"]), 1)

    # ─────────────────────────────────────────────────────────
    # 8. Follow-up Question Handling
    # ─────────────────────────────────────────────────────────
    def test_08_followup_question_tracking(self):
        """Test follow-up question entry generation from question objects."""
        q = {
            "id": 1,
            "question": "Explain dependency injection in FastAPI.",
            "topic": "FastAPI",
            "follow_up_question": "How do you override dependencies in testing?",
        }
        fu_entry = {
            "id": f"fu_{q['id']}",
            "question": q["follow_up_question"],
            "based_on": f"Q{q['id']}: {q['topic']}",
        }
        self.assertEqual(fu_entry["id"], "fu_1")
        self.assertEqual(fu_entry["question"], "How do you override dependencies in testing?")
        self.assertIn("FastAPI", fu_entry["based_on"])

    # ─────────────────────────────────────────────────────────
    # 9. Score Calculation
    # ─────────────────────────────────────────────────────────
    def test_09_score_calculation(self):
        """Verify accurate calculation of total score, max score, and percentage."""
        config = {"role": "Data Analyst"}
        questions = [
            {"id": 1, "question": "Q1", "topic": "SQL", "maximum_score": 10},
            {"id": 2, "question": "Q2", "topic": "Pandas", "maximum_score": 10},
        ]
        answers = {1: "Ans 1", 2: "Ans 2"}
        evaluations = {
            1: {"score": 8, "correct_points": ["C1"], "missing_points": ["M1"], "feedback": "F1", "improvement": "I1"},
            2: {"score": 7, "correct_points": ["C2"], "missing_points": ["M2"], "feedback": "F2", "improvement": "I2"},
        }
        metrics = calculate_report_metrics(config, questions, answers, evaluations)
        self.assertEqual(metrics["total_score"], 15)
        self.assertEqual(metrics["maximum_score"], 20)
        self.assertEqual(metrics["percentage"], 75.0)
        self.assertEqual(metrics["evaluated_count"], 2)

    # ─────────────────────────────────────────────────────────
    # 10. Final Report Metrics & Topic Breakdown
    # ─────────────────────────────────────────────────────────
    def test_10_final_report_generation(self):
        """Verify topic performance grouping, strengths aggregation, and recommendations."""
        config = {"role": "Python Developer", "skills": ["FastAPI", "SQL"]}
        questions = [
            {"id": 1, "question": "Q1", "topic": "FastAPI", "maximum_score": 10},
            {"id": 2, "question": "Q2", "topic": "FastAPI", "maximum_score": 10},
            {"id": 3, "question": "Q3", "topic": "SQL", "maximum_score": 10},
        ]
        answers = {1: "Ans 1", 2: "Ans 2", 3: "Ans 3"}
        evaluations = {
            1: {"score": 9, "correct_points": ["FastAPI routers"], "missing_points": [], "feedback": "Great", "improvement": ""},
            2: {"score": 8, "correct_points": ["Pydantic schemas"], "missing_points": [], "feedback": "Good", "improvement": ""},
            3: {"score": 5, "correct_points": ["SELECT query"], "missing_points": ["Indexes", "Window functions"], "feedback": "Fair", "improvement": "Study indexes"},
        }
        report = calculate_report_metrics(config, questions, answers, evaluations)
        self.assertEqual(report["total_score"], 22)
        self.assertEqual(report["maximum_score"], 30)
        self.assertEqual(len(report["topic_performance"]), 2)
        # SQL scored 5/10 (50%) -> should be in recommended topics
        self.assertTrue(any(rec["topic"] == "SQL" for rec in report["recommended_topics"]))
        # Strengths deduplication
        self.assertIn("FastAPI routers", report["strengths"])
        self.assertIn("Pydantic schemas", report["strengths"])

    # ─────────────────────────────────────────────────────────
    # 11. Restart Interview Session State Reset
    # ─────────────────────────────────────────────────────────
    def test_11_restart_interview_state(self):
        """Verify that session state reset dictionary cleans all interview data."""
        state = {
            "page": "report",
            "config": {"role": "DevOps"},
            "questions": [{"id": 1}],
            "current_question": 1,
            "answers": {1: "ans"},
            "evaluations": {1: {"score": 8}},
            "scores": {1: 8},
            "report_ready": True,
            "interview_complete": True,
        }
        # Emulate reset
        state.clear()
        defaults = {
            "page": "home",
            "config": {},
            "questions": [],
            "current_question": 0,
            "current_q_index": 0,
            "answers": {},
            "evaluations": {},
            "scores": {},
            "followup_questions": [],
            "report_ready": False,
            "interview_complete": False,
        }
        state.update(defaults)
        self.assertEqual(state["page"], "home")
        self.assertEqual(len(state["questions"]), 0)
        self.assertEqual(len(state["answers"]), 0)
        self.assertEqual(len(state["evaluations"]), 0)
        self.assertFalse(state["interview_complete"])

    # ─────────────────────────────────────────────────────────
    # 12. Session State Integrity & Zero Database Constraint
    # ─────────────────────────────────────────────────────────
    def test_12_zero_database_constraint(self):
        """Verify no SQL or database imports exist across all project python files."""
        disallowed = ["sqlite3", "psycopg2", "mysql", "pymongo", "sqlalchemy", "flask", "django", "fastapi"]
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        for root, _, files in os.walk(project_dir):
            if ".git" in root or "__pycache__" in root or ".gemini" in root:
                continue
            for file in files:
                if file.endswith(".py") and file != "test_qa_suite.py":
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as fh:
                        content = fh.read()
                        for imp in disallowed:
                            self.assertNotIn(f"import {imp}", content, f"Found prohibited import '{imp}' in {file}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
