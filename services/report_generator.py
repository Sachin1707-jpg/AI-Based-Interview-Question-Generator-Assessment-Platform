# ============================================================
# services/report_generator.py
# P_112 — AI-Based Interview Question Generator
# Generates and renders comprehensive, data-driven interview reports
# calculated strictly from actual session state evaluations.
# ============================================================

from typing import Any
import streamlit as st


# ─────────────────────────────────────────────────────────────
# REPORT DATA CALCULATION
# ─────────────────────────────────────────────────────────────
def calculate_report_metrics(
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    answers: dict[Any, str],
    evaluations: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute rigorous, 100% real statistics from actual session evaluations.
    No fake or hardcoded statistics are used.
    """
    total_score = 0
    total_max_score = 0
    question_breakdown: list[dict[str, Any]] = []
    topic_map: dict[str, dict[str, Any]] = {}
    all_correct_points: list[str] = []
    all_missing_points: list[str] = []
    all_improvements: list[str] = []

    for q in questions:
        q_id = q.get("id")
        q_text = q.get("question", "")
        topic = str(q.get("topic") or q.get("category") or "Technical Fundamentals").strip()
        max_q_score = int(q.get("maximum_score", 10))
        user_answer = str(answers.get(q_id, "")).strip()

        eval_data = evaluations.get(q_id, {})
        if not eval_data:
            continue

        q_score = int(eval_data.get("score", 0))
        total_score += q_score
        total_max_score += max_q_score

        # Correct & Missing points
        correct = eval_data.get("correct_points") or eval_data.get("correct_concepts") or eval_data.get("strengths") or []
        missing = eval_data.get("missing_points") or eval_data.get("missing_concepts") or []
        improvement = eval_data.get("improvement") or eval_data.get("improvement_suggestions") or ""
        feedback = eval_data.get("feedback", "Evaluation recorded.")

        if isinstance(correct, list):
            for c in correct:
                if str(c).strip():
                    all_correct_points.append(str(c).strip())

        if isinstance(missing, list):
            for m in missing:
                if str(m).strip():
                    all_missing_points.append(str(m).strip())

        if isinstance(improvement, list):
            for imp in improvement:
                if str(imp).strip():
                    all_improvements.append(str(imp).strip())
        elif isinstance(improvement, str) and improvement.strip():
            all_improvements.append(improvement.strip())

        # Question entry
        q_pct = round((q_score / max_q_score * 100), 1) if max_q_score > 0 else 0.0
        question_breakdown.append({
            "id": q_id,
            "question": q_text,
            "topic": topic,
            "difficulty": q.get("difficulty", "Medium"),
            "score": q_score,
            "maximum_score": max_q_score,
            "percentage": q_pct,
            "user_answer": user_answer if user_answer else "[No answer provided]",
            "expected_answer": q.get("expected_answer", ""),
            "rubric": q.get("rubric", []),
            "feedback": feedback,
            "correct_points": correct if isinstance(correct, list) else [],
            "missing_points": missing if isinstance(missing, list) else [],
            "improvement": improvement,
        })

        # Topic aggregation
        if topic not in topic_map:
            topic_map[topic] = {
                "score": 0,
                "max_score": 0,
                "question_count": 0,
            }
        topic_map[topic]["score"] += q_score
        topic_map[topic]["max_score"] += max_q_score
        topic_map[topic]["question_count"] += 1

    # Percentage and performance level
    percentage = round((total_score / total_max_score * 100), 1) if total_max_score > 0 else 0.0

    if percentage >= 85:
        performance_level = "Exceptional 🌟"
        summary_verdict = "Outstanding performance across all evaluated criteria. Demonstrated senior-level competence and clear conceptual mastery."
    elif percentage >= 70:
        performance_level = "Proficient ✅"
        summary_verdict = "Solid domain competence with minor gaps in edge-case coverage or deeper implementation nuance."
    elif percentage >= 50:
        performance_level = "Competent ⚠️"
        summary_verdict = "Foundational concepts are present, but key technical details, best practices, or depth require revision."
    else:
        performance_level = "Needs Development ❌"
        summary_verdict = "Core conceptual gaps identified. Structured study and hands-on practice are strongly recommended."

    # Process topic performance
    topic_performance = []
    recommended_topics = []
    for t_name, t_data in topic_map.items():
        t_pct = round((t_data["score"] / t_data["max_score"] * 100), 1) if t_data["max_score"] > 0 else 0.0
        topic_entry = {
            "topic": t_name,
            "score": t_data["score"],
            "max_score": t_data["max_score"],
            "percentage": t_pct,
            "question_count": t_data["question_count"],
        }
        topic_performance.append(topic_entry)
        if t_pct < 75:
            recommended_topics.append({
                "topic": t_name,
                "percentage": t_pct,
                "score_str": f"{t_data['score']}/{t_data['max_score']}",
            })

    # Sort topics
    topic_performance.sort(key=lambda x: x["percentage"], reverse=True)
    recommended_topics.sort(key=lambda x: x["percentage"])

    # Deduplicate strengths & improvements
    unique_strengths = list(dict.fromkeys(all_correct_points))
    unique_improvements = list(dict.fromkeys(all_missing_points + all_improvements))

    return {
        "total_score": total_score,
        "maximum_score": total_max_score,
        "percentage": percentage,
        "performance_level": performance_level,
        "summary_verdict": summary_verdict,
        "question_breakdown": question_breakdown,
        "topic_performance": topic_performance,
        "strengths": unique_strengths,
        "areas_for_improvement": unique_improvements,
        "recommended_topics": recommended_topics,
        "evaluated_count": len(question_breakdown),
        "total_questions": len(questions),
    }


# ─────────────────────────────────────────────────────────────
# STREAMLIT REPORT RENDERER
# ─────────────────────────────────────────────────────────────
def render_interview_report():
    """
    Renders the complete, highly professional interview report interface
    using only native Streamlit components.
    """
    st.title("📊 Final Interview Performance Report")
    st.caption("Comprehensive assessment results generated from your AI evaluation session.")

    config = st.session_state.get("config", {})
    questions = st.session_state.get("questions", [])
    answers = st.session_state.get("answers", {})
    evaluations = st.session_state.get("evaluations", {})

    # ── Empty State ──
    if not evaluations:
        with st.container(border=True):
            st.subheader("📋 No Evaluations Found")
            st.info("You haven't completed any interview questions yet. Start an interview session to generate your final performance report.")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("💬 Go to Interview", use_container_width=True, type="primary"):
                    st.session_state.page = "interview"
                    st.rerun()
            with c2:
                if st.button("⚙️ Go to Configuration", use_container_width=True):
                    st.session_state.page = "configure"
                    st.rerun()
        return

    # Calculate metrics
    report = calculate_report_metrics(config, questions, answers, evaluations)

    # ── 1. Candidate Profile Summary ──
    with st.container(border=True):
        st.subheader("👤 Candidate & Session Profile")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        with p1:
            st.metric("Job Role", config.get("role", "N/A"))
        with p2:
            st.metric("Experience", config.get("experience", "N/A"))
        with p3:
            st.metric("Interview Type", config.get("interview_type", "N/A"))
        with p4:
            st.metric("Difficulty", config.get("difficulty", "N/A"))
        with p5:
            st.metric("Questions", f"{report['evaluated_count']} / {report['total_questions']}")
        with p6:
            skills = config.get("skills", [])
            st.metric("Skills Tested", len(skills) if skills else "General")

    st.write("")

    # ── 2. Executive Score Summary ──
    with st.container(border=True):
        st.subheader("🏆 Executive Score Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(
                label="Total Score",
                value=f"{report['total_score']} / {report['maximum_score']}",
                delta=f"{report['percentage']:.0f}% Achieved",
            )
        with s2:
            st.metric(label="Percentage", value=f"{report['percentage']:.1f}%")
        with s3:
            st.metric(label="Performance Rating", value=report["performance_level"])
        with s4:
            st.metric(
                label="Questions Completed",
                value=f"{report['evaluated_count']} / {report['total_questions']}",
            )

        prog_val = min(1.0, max(0.0, report["percentage"] / 100.0))
        st.progress(prog_val, text=f"Overall Mastery: {report['total_score']} / {report['maximum_score']} points ({report['percentage']}%)")

        st.info(f"**Strategic Verdict:** {report['summary_verdict']}")

    st.write("")

    # ── 3. Topic & Skill Performance Breakdown ──
    with st.container(border=True):
        st.subheader("📈 Skill & Topic Performance Breakdown")
        if report["topic_performance"]:
            cols = st.columns(min(3, max(1, len(report["topic_performance"]))))
            for i, tp in enumerate(report["topic_performance"]):
                col = cols[i % len(cols)]
                with col:
                    with st.container(border=True):
                        st.write(f"**{tp['topic']}**")
                        st.caption(f"{tp['question_count']} question{'s' if tp['question_count'] > 1 else ''} evaluated")
                        st.metric("Topic Score", f"{tp['score']} / {tp['max_score']}", delta=f"{tp['percentage']}%")
                        tp_prog = min(1.0, max(0.0, tp["percentage"] / 100.0))
                        st.progress(tp_prog)
        else:
            st.caption("No specific topic breakdown available.")

    st.write("")

    # ── 4. Strengths & Areas for Improvement ──
    col_str, col_imp = st.columns(2)

    with col_str:
        with st.container(border=True):
            st.subheader("✅ Demonstrated Strengths")
            if report["strengths"]:
                for item in report["strengths"][:8]:
                    st.write(f"• **{item}**")
            else:
                st.caption("No specific technical strengths recorded.")

    with col_imp:
        with st.container(border=True):
            st.subheader("⚠️ Key Areas for Improvement")
            if report["areas_for_improvement"]:
                for item in report["areas_for_improvement"][:8]:
                    st.write(f"• {item}")
            else:
                st.caption("No critical improvement areas identified.")

    st.write("")

    # ── 5. Recommended Topics for Preparation ──
    with st.container(border=True):
        st.subheader("📚 Recommended Topics to Review")
        if report["recommended_topics"]:
            st.write("Based on specific rubric criteria where full points were not awarded, prioritize these topics:")
            for rec in report["recommended_topics"]:
                st.warning(f"📌 **{rec['topic']}** (Current Score: **{rec['score_str']}** · **{rec['percentage']}%**) — Focus on core architecture, trade-offs, and practical edge-case implementations.")
        else:
            st.success("🌟 **Great technical coverage!** You achieved strong marks (75%+) across all tested topics. Continue practicing advanced architectural scalability.")

    st.write("")

    # ── 6. Question-by-Question Detailed Analysis ──
    with st.container(border=True):
        st.subheader("📋 Question-by-Question Deep Dive")
        st.caption("Expand any question to inspect candidate answer, rubric breakdown, and AI feedback.")

        for q_data in report["question_breakdown"]:
            score_val = q_data["score"]
            max_val = q_data["maximum_score"]
            pct_val = q_data["percentage"]

            badge = "🟢" if pct_val >= 80 else "🟡" if pct_val >= 50 else "🔴"
            title_preview = q_data["question"][:65] + "..." if len(q_data["question"]) > 65 else q_data["question"]
            expander_title = f"{badge} Q{q_data['id']}: {title_preview}  |  {score_val}/{max_val} pts ({pct_val}%)"

            with st.expander(expander_title, expanded=False):
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Score", f"{score_val} / {max_val}")
                with m2:
                    st.write(f"**Topic:** {q_data['topic']}")
                with m3:
                    st.write(f"**Difficulty:** {q_data['difficulty']}")

                st.info(f"**Question:**\n\n{q_data['question']}")
                st.divider()

                c_ans, c_fb = st.columns(2)
                with c_ans:
                    st.markdown("**Candidate's Answer:**")
                    st.write(q_data["user_answer"])

                with c_fb:
                    st.markdown("**AI Evaluator Feedback:**")
                    st.write(q_data["feedback"])

                    if q_data["correct_points"]:
                        st.write("**Correct Points Demonstrated:**")
                        for cp in q_data["correct_points"]:
                            st.write(f"✅ {cp}")

                    if q_data["missing_points"]:
                        st.write("**Missing / Incomplete Concepts:**")
                        for mp in q_data["missing_points"]:
                            st.write(f"⚠️ {mp}")

                    if q_data["improvement"]:
                        st.write(f"💡 **Improvement:** {q_data['improvement']}")

                if q_data["rubric"]:
                    st.divider()
                    st.caption("**Assessment Rubric Criteria:**")
                    if isinstance(q_data["rubric"], list):
                        for r in q_data["rubric"]:
                            if isinstance(r, dict):
                                st.caption(f"- {r.get('criterion', '')} ({r.get('points', '')} pts)")
                            else:
                                st.caption(f"- {r}")

    st.write("")

    # ── 7. Actions: Start New Interview & Export ──
    with st.container(border=True):
        st.subheader("🎯 Next Actions")
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            if st.button("🔄 Start New Interview", use_container_width=True, type="primary"):
                st.session_state.clear()
                st.session_state.page = "home"
                st.rerun()

        with act_col2:
            report_lines = [
                "=" * 65,
                "AI INTERVIEW REPORT — P_112",
                "=" * 65,
                f"Role:              {config.get('role', 'N/A')}",
                f"Experience:        {config.get('experience', 'N/A')}",
                f"Interview Type:    {config.get('interview_type', 'N/A')}",
                f"Difficulty:        {config.get('difficulty', 'N/A')}",
                f"Skills:            {', '.join(config.get('skills', []))}",
                "",
                f"Total Score:       {report['total_score']} / {report['maximum_score']}",
                f"Percentage:        {report['percentage']:.1f}%",
                f"Performance Level: {report['performance_level']}",
                "",
                "=" * 65,
                "TOPIC PERFORMANCE",
                "=" * 65,
            ]
            for tp in report["topic_performance"]:
                report_lines.append(f"- {tp['topic']}: {tp['score']}/{tp['max_score']} ({tp['percentage']}%)")

            report_lines += [
                "",
                "=" * 65,
                "QUESTION BREAKDOWN",
                "=" * 65,
            ]
            for qd in report["question_breakdown"]:
                report_lines += [
                    f"\nQ{qd['id']}: {qd['question']}",
                    f"Topic: {qd['topic']} | Score: {qd['score']}/{qd['maximum_score']} ({qd['percentage']}%)",
                    f"Candidate Answer: {qd['user_answer']}",
                    f"Feedback: {qd['feedback']}",
                ]
                if qd["correct_points"]:
                    report_lines.append(f"Correct: {', '.join(qd['correct_points'])}")
                if qd["missing_points"]:
                    report_lines.append(f"Missing: {', '.join(qd['missing_points'])}")
                if qd["improvement"]:
                    report_lines.append(f"Improvement: {qd['improvement']}")

            report_lines += [
                "",
                "=" * 65,
                "Generated by AI-Based Interview Question Generator — P_112",
                "=" * 65,
            ]
            report_text = "\n".join(report_lines)

            file_role = str(config.get("role", "candidate")).lower().replace(" ", "_")
            st.download_button(
                label="📥 Download Full Report (.txt)",
                data=report_text,
                file_name=f"interview_report_{file_role}.txt",
                mime="text/plain",
                use_container_width=True,
            )
