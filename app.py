# ============================================================
# P_112 — AI-Based Interview Question Generator
# app.py — Pure Streamlit Interface  |  OpenAI GPT-4o Integration
# ============================================================

import os
import streamlit as st
import time
from dotenv import load_dotenv

from services.question_generator import generate_questions
from services.evaluator import evaluate_answer
from services.report_generator import render_interview_report

# ─────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY", "")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Question Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
def init_state():
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
        "gen_warnings": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# ─────────────────────────────────────────
# STATIC DATA & OPTIONS
# ─────────────────────────────────────────
ROLE_OPTIONS = [
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Python Developer",
    "Data Analyst",
    "ML Engineer",
    "DevOps Engineer",
    "Custom",
]

EXPERIENCE_OPTIONS = [
    "Fresher",
    "0-1 years",
    "1-3 years",
    "3-5 years",
    "5+ years",
]

INTERVIEW_TYPE_OPTIONS = [
    "Technical",
    "Behavioral",
    "Coding",
    "System Design",
    "Mixed",
]

DIFFICULTY_OPTIONS = ["Easy", "Medium", "Hard"]

CATEGORY_OPTIONS = [
    "Technical",
    "Conceptual",
    "Scenario Based",
    "Problem Solving",
    "Behavioral",
]

SKILLS_OPTIONS = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "SQL",
    "HTML", "CSS", "React", "Vue.js", "Angular", "Next.js", "Tailwind CSS",
    "FastAPI", "Django", "Flask", "Node.js", "Express.js", "REST API", "GraphQL",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Analysis", "Feature Engineering", "Statistics", "MLOps",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "LLMs",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Linux",
    "System Design", "Data Structures", "Algorithms", "Git", "Agile / Scrum",
]


# ─────────────────────────────────────────
# NAVIGATION HELPER
# ─────────────────────────────────────────
def go_to(page: str):
    st.session_state.page = page
    st.rerun()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🎯 AI Interviewer")
        st.caption("AI-Based Interview Question Generator")

        st.divider()

        # Step-based Page Navigation
        st.subheader("Navigation")
        pages = [
            ("home", "🏠 1. Home"),
            ("configure", "⚙️ 2. Configuration"),
            ("interview", "💬 3. AI Interview"),
            ("report", "📊 4. Final Report"),
        ]

        for key, label in pages:
            is_active = st.session_state.page == key
            btn_style = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_style):
                go_to(key)

        st.divider()

        # Session Status Card
        with st.container(border=True):
            st.subheader("📌 Session Tracker")
            q_list = st.session_state.get("questions", [])
            evals = st.session_state.get("evaluations", {})
            cfg = st.session_state.get("config", {})

            if q_list:
                role_label = cfg.get("role", "General")
                st.write(f"**Role:** {role_label}")
                st.write(f"**Progress:** {len(evals)} / {len(q_list)} evaluated")
                prog_pct = min(1.0, len(evals) / len(q_list)) if q_list else 0.0
                st.progress(prog_pct)
            else:
                st.caption("No active interview session. Click **Configuration** to set up.")

        st.write("")

        # API Status Card
        with st.container(border=True):
            st.subheader("🔑 Engine Status")
            if GEMINI_API_KEY:
                st.success("✅ Google Gemini Connected")
            else:
                st.error("❌ API Key Missing")
                st.caption("Add GEMINI_API_KEY to your `.env` file")

        st.write("")

        if st.button("🔄 Reset / Start Over", use_container_width=True):
            st.session_state.clear()
            init_state()
            st.rerun()


# ─────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────
def page_home():
    # Hero Section
    with st.container(border=True):
        st.title("🎯 AI-Based Interview Question Generator")
        st.subheader("AI-Powered Role-Specific Interview Preparation & Evaluation")
        st.write(
            "Generate authentic, role-aligned technical interview questions, practice in a structured "
            "live simulation, and receive objective rubric-based evaluations with comprehensive performance analytics."
        )
        st.write("")
        b_start, _ = st.columns([1, 2])
        with b_start:
            if st.button("🚀 Start Interview Configuration", use_container_width=True, type="primary"):
                go_to("configure")

    st.write("")

    # Feature Grid
    st.subheader("✨ Core Capabilities")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        with st.container(border=True):
            st.markdown("#### 🧠 Smart Generation")
            st.write("Generates customized technical, scenario-based, and problem-solving questions tailored to exact roles and experience.")

    with f2:
        with st.container(border=True):
            st.markdown("#### 💬 Live Simulation")
            st.write("Simulates actual candidate technical interview rounds with one-question-at-a-time pacing and answer drafting.")

    with f3:
        with st.container(border=True):
            st.markdown("#### ⚖️ Rubric Evaluation")
            st.write("Evaluates candidate answers objectively against multi-criteria rubrics without personal bias or hallucinated metrics.")

    with f4:
        with st.container(border=True):
            st.markdown("#### 📊 Analytical Reports")
            st.write("Calculates real topic breakdowns, strengths, missing concepts, score distributions, and recommended study areas.")

    st.write("")

    # Workflow Steps
    with st.container(border=True):
        st.subheader("🔄 How It Works")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(label="Step 1", value="⚙️ Configure")
            st.caption("Select your role, experience level, tech stack, and difficulty.")
        with s2:
            st.metric(label="Step 2", value="🧠 Generate")
            st.caption("AI generates targeted questions with model answers and scoring rubrics.")
        with s3:
            st.metric(label="Step 3", value="📝 Practice")
            st.caption("Answer questions one by one and receive real-time AI assessments.")
        with s4:
            st.metric(label="Step 4", value="📈 Report")
            st.caption("Review your executive score, topic mastery, and downloadable report.")

    st.write("")

    # Supported Domains
    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.subheader("🎯 Supported Interview Types")
            st.write("• **Technical Interviews**: Deep dive into languages, frameworks, and tools.")
            st.write("• **Coding & Algorithms**: Data structures, complexity, and problem solving.")
            st.write("• **System Design**: Architectural trade-offs, scalability, and APIs.")
            st.write("• **Behavioral**: Scenario-based teamwork, leadership, and STAR responses.")
            st.write("• **Mixed Mode**: Well-rounded combination across all core areas.")

    with col_right:
        with st.container(border=True):
            st.subheader("💼 Supported Technical Roles")
            st.write("• Python Developer · Backend Engineer · Frontend Developer")
            st.write("• Full Stack Developer · Machine Learning Engineer · Data Analyst")
            st.write("• DevOps / Cloud Engineer · AI / LLM Engineer")
            st.write("• **Custom Roles**: Fully supported via custom role configuration.")


# ─────────────────────────────────────────
# VALIDATION HELPER
# ─────────────────────────────────────────
def _validate_config(role, custom_role, experience, skills, custom_skills_raw,
                    interview_type, difficulty, num_questions, categories):
    errors = []
    warnings = []

    # Role
    if role == "Custom":
        if not custom_role.strip():
            errors.append("Custom role name cannot be empty when 'Custom' is selected.")
        elif len(custom_role.strip()) < 3:
            errors.append("Custom role name must be at least 3 characters.")
        elif len(custom_role.strip()) > 80:
            errors.append("Custom role name must be 80 characters or fewer.")

    # Skills
    all_skills = list(skills)
    if custom_skills_raw.strip():
        extra = [s.strip() for s in custom_skills_raw.split(",") if s.strip()]
        too_long = [s for s in extra if len(s) > 50]
        if too_long:
            errors.append(f"These custom skills exceed 50 characters: {', '.join(too_long)}")
        all_skills.extend(extra)

    if not all_skills:
        errors.append("Please select at least one skill or enter custom skills.")
    elif len(all_skills) > 15:
        warnings.append(f"You selected {len(all_skills)} skills. For best focus, 5–10 skills are recommended.")

    # Categories
    if not categories:
        errors.append("Please select at least one Question Category.")

    # Question count
    if not (5 <= num_questions <= 15):
        errors.append("Number of questions must be between 5 and 15.")

    return errors, warnings, all_skills


# ─────────────────────────────────────────
# PAGE 2 — INTERVIEW CONFIGURATION
# ─────────────────────────────────────────
def page_configure():
    st.title("⚙️ Interview Configuration")
    st.caption("Customize role profile, skills, difficulty, and question distribution.")

    # Notice if existing session is present
    if st.session_state.get("config"):
        prev = st.session_state.config
        with st.container(border=True):
            st.info(
                f"ℹ️ **Current Session Active:** Role: **{prev.get('role', '—')}** · "
                f"Type: **{prev.get('interview_type', '—')}** · Difficulty: **{prev.get('difficulty', '—')}**"
            )

    with st.form("config_form", clear_on_submit=False):
        # Section 1: Role & Experience
        with st.container(border=True):
            st.subheader("1️⃣ Role & Experience Level")
            r1, r2 = st.columns(2)
            with r1:
                role_choice = st.selectbox(
                    "Job Role *",
                    options=ROLE_OPTIONS,
                    index=0,
                    help="Select the target role or choose Custom.",
                )
                custom_role = ""
                if role_choice == "Custom":
                    custom_role = st.text_input(
                        "Enter Custom Role Title *",
                        placeholder="e.g. LLM Systems Engineer",
                    )
            with r2:
                experience = st.selectbox(
                    "Experience Level *",
                    options=EXPERIENCE_OPTIONS,
                    index=1,
                    help="Select candidate experience bracket.",
                )

        st.write("")

        # Section 2: Skills & Tech Stack
        with st.container(border=True):
            st.subheader("2️⃣ Skills & Tech Stack")
            skills = st.multiselect(
                "Select Skills *",
                options=SKILLS_OPTIONS,
                default=["Python", "FastAPI", "REST API"],
                help="Select primary tools and technologies.",
            )
            custom_skills_raw = st.text_input(
                "Additional Custom Skills (comma-separated)",
                placeholder="e.g. Celery, Redis, SQLAlchemy",
                help="Add any specialized technologies not in the standard list.",
            )

        st.write("")

        # Section 3: Interview Type & Difficulty
        with st.container(border=True):
            st.subheader("3️⃣ Interview Type & Difficulty")
            t1, t2 = st.columns(2)
            with t1:
                interview_type = st.selectbox(
                    "Interview Type *",
                    options=INTERVIEW_TYPE_OPTIONS,
                    index=0,
                )
            with t2:
                difficulty = st.selectbox(
                    "Difficulty Level *",
                    options=DIFFICULTY_OPTIONS,
                    index=1,
                )

        st.write("")

        # Section 4: Question Count & Categories
        with st.container(border=True):
            st.subheader("4️⃣ Question Settings & Categories")
            q_col1, q_col2 = st.columns([2, 1])
            with q_col1:
                num_questions = st.slider(
                    "Number of Questions *",
                    min_value=5,
                    max_value=15,
                    value=5,
                    step=1,
                )
            with q_col2:
                st.metric("Estimated Duration", f"{num_questions * 3}–{num_questions * 5} min")

            st.write("**Question Categories *:**")
            categories = st.multiselect(
                "Categories",
                options=CATEGORY_OPTIONS,
                default=["Technical", "Scenario Based"],
                label_visibility="collapsed",
            )

        st.write("")

        # Submit Button
        submitted = st.form_submit_button(
            "🤖 Generate Interview Questions",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            errors, warnings, all_skills = _validate_config(
                role_choice, custom_role, experience,
                skills, custom_skills_raw,
                interview_type, difficulty,
                num_questions, categories,
            )

            for w in warnings:
                st.warning(f"⚠️ {w}")

            if errors:
                with st.container(border=True):
                    st.error("Please resolve the following configuration issues:")
                    for err in errors:
                        st.write(f"• ❌ {err}")
            else:
                final_role = custom_role.strip() if role_choice == "Custom" else role_choice

                interview_config = {
                    "role":           final_role,
                    "experience":     experience,
                    "skills":         all_skills,
                    "interview_type": interview_type,
                    "difficulty":     difficulty,
                    "question_count": num_questions,
                    "categories":     categories,
                }

                # Reset state for fresh interview
                st.session_state.config             = interview_config
                st.session_state.current_question   = 0
                st.session_state.current_q_index    = 0
                st.session_state.answers            = {}
                st.session_state.evaluations        = {}
                st.session_state.scores             = {}
                st.session_state.interview_complete = False
                st.session_state.report_ready       = False
                st.session_state.followup_questions = []

                with st.spinner(f"🧠 Generating {num_questions} {difficulty} questions for {final_role}..."):
                    try:
                        questions, gen_warnings = generate_questions(interview_config)
                        st.session_state.questions = questions
                        st.session_state.gen_warnings = gen_warnings
                    except EnvironmentError:
                        st.error("❌ GEMINI_API_KEY is not configured in `.env`.")
                        st.stop()
                    except RuntimeError as e:
                        st.error(f"❌ API connection issue: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Question generation error: {e}")
                        st.stop()

                actual_count = len(st.session_state.questions)
                st.success(f"✅ Generated {actual_count} questions for **{final_role}**! Starting interview...")
                time.sleep(0.5)
                go_to("interview")


# ─────────────────────────────────────────
# PAGE 3 — AI INTERVIEW
# ─────────────────────────────────────────
def page_interview():
    st.title("💬 AI Interview Session")
    st.caption("Answer each question sequentially. Your answers are evaluated immediately by AI.")

    config = st.session_state.get("config", {})
    questions = st.session_state.get("questions", [])
    answers = st.session_state.get("answers", {})
    evaluations = st.session_state.get("evaluations", {})
    scores = st.session_state.get("scores", {})

    # ── Empty State ──
    if not questions:
        with st.container(border=True):
            st.subheader("⚠️ No Active Interview Session")
            st.info("No generated questions found. Please configure your interview parameters first.")
            if st.button("⚙️ Go to Configuration", type="primary"):
                go_to("configure")
        return

    # Synchronize index
    idx = st.session_state.get("current_question", 0)
    if idx >= len(questions):
        idx = max(0, len(questions) - 1)
    st.session_state.current_question = idx
    st.session_state.current_q_index = idx

    # Header Bar
    with st.container(border=True):
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Target Role", config.get("role", "N/A"))
        with h2:
            st.metric("Interview Type", config.get("interview_type", "N/A"))
        with h3:
            st.metric("Difficulty", config.get("difficulty", "N/A"))
        with h4:
            st.metric("Evaluated", f"{len(evaluations)} / {len(questions)}")

    for w in st.session_state.get("gen_warnings", []):
        st.info(f"💡 {w}")

    # Interview Tabs
    tab_active, tab_review, tab_followup = st.tabs(
        ["📝 Active Question", "🔍 Review All Questions", "🔄 Follow-up Questions"]
    )

    # ── Tab 1: Active Single-Question Flow ──
    with tab_active:
        q = questions[idx]
        q_id = q.get("id", idx + 1)
        total_q = len(questions)

        pct_done = ((idx + 1) / total_q) * 100
        st.progress((idx + 1) / total_q, text=f"Question {idx + 1} of {total_q} ({pct_done:.0f}%)")

        # Question Card
        with st.container(border=True):
            qc1, qc2, qc3 = st.columns([2, 1, 1])
            with qc1:
                st.subheader(f"Question {idx + 1} of {total_q}")
            with qc2:
                st.write(f"**Topic:** {q.get('topic') or q.get('category', 'Technical')}")
            with qc3:
                st.write(f"**Difficulty:** {q.get('difficulty', 'Medium')}")

            st.info(f"**{q.get('question', '')}**")

            with st.expander("💡 Assessment Rubric & Criteria", expanded=False):
                rubric_val = q.get("rubric", [])
                if isinstance(rubric_val, list):
                    for r in rubric_val:
                        if isinstance(r, dict):
                            st.write(f"• **{r.get('criterion', '')}**: {r.get('points', 1)} points")
                        else:
                            st.write(f"• {r}")
                else:
                    st.write(str(rubric_val))

            st.divider()

            # Answer Area
            is_evaluated = q_id in evaluations
            existing_ans = answers.get(q_id, "")

            user_ans = st.text_area(
                "Candidate Answer",
                value=existing_ans,
                height=180,
                placeholder="Type your technical response here. Include reasoning, syntax, concepts, and architectural details...",
                key=f"ans_text_{q_id}",
                disabled=is_evaluated,
            )

            # State A: Not evaluated yet
            if not is_evaluated:
                b_prev, b_save, b_sub = st.columns([1, 1, 2])

                with b_prev:
                    if idx > 0:
                        if st.button("⬅️ Previous", use_container_width=True):
                            if user_ans.strip():
                                st.session_state.answers[q_id] = user_ans.strip()
                            st.session_state.current_question = idx - 1
                            st.rerun()

                with b_save:
                    if st.button("💾 Save Draft", use_container_width=True):
                        if user_ans.strip():
                            st.session_state.answers[q_id] = user_ans.strip()
                            st.success("Draft saved!")
                        else:
                            st.warning("Type an answer to save a draft.")

                with b_sub:
                    if st.button("🚀 Submit Answer", use_container_width=True, type="primary"):
                        if not user_ans.strip():
                            st.error("Please enter an answer before submitting.")
                        else:
                            with st.spinner("🤖 AI Evaluator is analyzing your response against the rubric..."):
                                try:
                                    eval_result = evaluate_answer(
                                        question=q.get("question", ""),
                                        expected_answer=q.get("expected_answer", ""),
                                        rubric=q.get("rubric", []),
                                        candidate_answer=user_ans.strip(),
                                        maximum_score=q.get("maximum_score", 10),
                                    )
                                except Exception as e:
                                    st.error(f"❌ Evaluation error: {e}")
                                    st.stop()

                            st.session_state.answers[q_id] = user_ans.strip()
                            st.session_state.evaluations[q_id] = eval_result
                            st.session_state.scores[q_id] = eval_result.get("score", 0)

                            # Capture follow-up if present
                            fu_q = q.get("follow_up_question")
                            if fu_q:
                                fu_entry = {
                                    "id": f"fu_{q_id}",
                                    "question": fu_q,
                                    "based_on": f"Q{idx + 1}: {q.get('topic', 'Topic')}",
                                }
                                if not any(f["id"] == fu_entry["id"] for f in st.session_state.followup_questions):
                                    st.session_state.followup_questions.append(fu_entry)

                            if len(st.session_state.evaluations) == total_q:
                                st.session_state.interview_complete = True
                                st.session_state.report_ready = True

                            st.rerun()

            # State B: Evaluated
            else:
                eval_data = evaluations[q_id]
                score_num = eval_data.get("score", 0)
                max_num = q.get("maximum_score", 10)

                with st.container(border=True):
                    st.subheader("🎯 AI Evaluation Result")
                    er1, er2, er3 = st.columns(3)
                    with er1:
                        st.metric("Score", f"{score_num} / {max_num}")
                    with er2:
                        pct = (score_num / max_num * 100) if max_num > 0 else 0
                        st.metric("Percentage", f"{pct:.0f}%")
                    with er3:
                        st.metric("Status", "Evaluated ✅")

                    st.markdown(f"**Feedback:** {eval_data.get('feedback', '')}")

                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.write("**Correct Concepts Demonstrated:**")
                        for cp in eval_data.get("correct_points", []):
                            st.write(f"✅ {cp}")
                        if not eval_data.get("correct_points"):
                            st.caption("No specific correct points identified.")

                    with c_right:
                        st.write("**Missing / Incomplete Concepts:**")
                        for mp in eval_data.get("missing_points", []):
                            st.write(f"⚠️ {mp}")
                        if not eval_data.get("missing_points"):
                            st.caption("No missing concepts flagged.")

                    if eval_data.get("improvement"):
                        st.write(f"💡 **Improvement Advice:** {eval_data['improvement']}")

                    with st.expander("📖 Model Expected Answer", expanded=False):
                        st.write(q.get("expected_answer", "N/A"))

                st.write("")

                # Navigation Buttons
                nav1, nav2, nav3 = st.columns([1, 1, 2])
                with nav1:
                    if idx > 0:
                        if st.button("⬅️ Previous Question", use_container_width=True):
                            st.session_state.current_question = idx - 1
                            st.rerun()

                with nav2:
                    if st.button("🔄 Retake Question", use_container_width=True):
                        st.session_state.evaluations.pop(q_id, None)
                        st.session_state.scores.pop(q_id, None)
                        st.session_state.interview_complete = False
                        st.rerun()

                with nav3:
                    is_last = (idx == total_q - 1)
                    if is_last or len(evaluations) == total_q:
                        if st.button("📊 View Final Report", use_container_width=True, type="primary"):
                            st.session_state.interview_complete = True
                            st.session_state.report_ready = True
                            go_to("report")
                    else:
                        if st.button("Next Question ➡️", use_container_width=True, type="primary"):
                            st.session_state.current_question = idx + 1
                            st.rerun()

    # ── Tab 2: Review All ──
    with tab_review:
        st.subheader("📋 Complete Questions Overview")
        if not answers and not evaluations:
            st.info("No answers submitted yet.")
        else:
            for q_item in questions:
                qid = q_item.get("id")
                ev = evaluations.get(qid)
                badge = f"Score: {ev['score']}/10" if ev else "Pending"

                with st.expander(f"Q{qid}: {q_item.get('question', '')[:65]}... — [{badge}]", expanded=False):
                    st.write("**Question:**", q_item.get("question", ""))
                    st.write(f"**Topic:** {q_item.get('topic', 'N/A')} · **Difficulty:** {q_item.get('difficulty', 'N/A')}")
                    st.divider()
                    st.write("**Candidate Answer:**")
                    st.write(answers.get(qid, "[No answer]"))

                    if ev:
                        st.divider()
                        st.write(f"**Score:** {ev.get('score', 0)} / {q_item.get('maximum_score', 10)}")
                        st.write(f"**Feedback:** {ev.get('feedback', '')}")

    # ── Tab 3: Follow-up Questions ──
    with tab_followup:
        st.subheader("🔄 Adaptive Follow-up Questions")
        followups = st.session_state.get("followup_questions", [])
        if not followups:
            st.info("Follow-up questions generated by AI will appear here as you submit answers.")
        else:
            for fq in followups:
                with st.container(border=True):
                    st.write(f"**{fq.get('based_on', 'Follow-up')}:** {fq.get('question', '')}")
                    fu_key = f"fu_ans_{fq.get('id')}"
                    fu_text = st.text_area(
                        "Your Follow-up Response",
                        value=st.session_state.get(fu_key, ""),
                        height=100,
                        key=f"input_{fu_key}",
                    )
                    if st.button("Save Follow-up", key=f"btn_{fq.get('id')}"):
                        if fu_text.strip():
                            st.session_state[fu_key] = fu_text.strip()
                            st.success("✅ Follow-up saved!")


# ─────────────────────────────────────────
# PAGE 4 — FINAL REPORT
# ─────────────────────────────────────────
def page_report():
    render_interview_report()


# ─────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────
def main():
    render_sidebar()
    current_page = st.session_state.page
    if current_page == "home":
        page_home()
    elif current_page == "configure":
        page_configure()
    elif current_page == "interview":
        page_interview()
    elif current_page == "report":
        page_report()
    else:
        page_home()


if __name__ == "__main__":
    main()
