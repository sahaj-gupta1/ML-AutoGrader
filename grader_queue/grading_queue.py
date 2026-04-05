import json
import threading
from datetime import datetime


def _grade_all_worker(app, assignment_id: int):
    with app.app_context():
        from models import db, Assignment, Submission

        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return

        assignment.grading_status = "grading"
        db.session.commit()

        rubric       = assignment.rubric
        task_type    = assignment.task_type
        dataset_path = assignment.dataset_path

        # Extract teacher custom subtasks from rubric if present
        teacher_custom = {}
        for step, info in rubric.items():
            if "custom_subtasks" in info and info["custom_subtasks"]:
                teacher_custom[step] = info["custom_subtasks"]

        submissions = Submission.query.filter_by(
            assignment_id = assignment_id,
            status        = "submitted",
        ).all()

        if not submissions:
            assignment.grading_status = "completed"
            db.session.commit()
            return

        print(f"\n[Queue] Grading '{assignment.title}' — {len(submissions)} submission(s)")

        for i, sub in enumerate(submissions, start=1):
            print(f"[Queue] {i}/{len(submissions)} — student_id={sub.student_id}")
            sub.status = "grading"
            db.session.commit()

            try:
                from core.grader_pipeline import run_grader

                result = run_grader(
                    notebook_path           = sub.file_path,
                    rubric                  = rubric,
                    task_type               = task_type,
                    dataset_path            = dataset_path,
                    teacher_custom_subtasks = teacher_custom,
                )

                if result["status"] == "error":
                    raise Exception(result.get("error_message", "Unknown error"))

                sub.status      = "graded"
                sub.score       = result["final_total_score"]
                sub.max_score   = result["max_score"]
                sub.result_json = json.dumps(result)
                sub.graded_at   = datetime.utcnow()
                sub.released    = True

                print(f"[Queue]   Score: {sub.score}/{sub.max_score}")

            except Exception as e:
                print(f"[Queue]   ERROR: {e}")
                import traceback
                traceback.print_exc()

                max_score = float(sum(v["points"] for v in rubric.values()))
                sub.status      = "graded"
                sub.score       = 0.0
                sub.max_score   = max_score
                sub.released    = True
                sub.result_json = json.dumps({
                    "status":             "error",
                    "final_total_score":  0,
                    "max_score":          max_score,
                    "extracted_timeline": [],
                    "step_detail":        {},
                    "final_scores":       {},
                    "system_penalties":   [],
                    "missed_steps":       list(rubric.keys()),
                    "feedback": {
                        "strengths":    [],
                        "weaknesses":   [f"Grading failed: {e}"],
                        "improvements": ["Please resubmit your notebook."],
                    }
                })

            db.session.commit()

        assignment.grading_status = "completed"
        db.session.commit()
        print(f"[Queue] Done — '{assignment.title}' results released.")


def start_grading(app, assignment_id: int):
    thread = threading.Thread(
        target = _grade_all_worker,
        args   = (app, assignment_id),
        daemon = True,
        name   = f"grader-{assignment_id}",
    )
    thread.start()
    return thread
