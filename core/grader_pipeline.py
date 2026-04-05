"""
Grader Pipeline
===============
Orchestrates the full grading flow:
  1. Parse .ipynb → extract code
  2. Run enhanced AST analysis (sub-task level detection)
  3. Score proportionally with ordering penalties
  4. Call Ollama once for detailed feedback
"""

import csv
import os
from datetime import datetime

from core.ipynb_parser   import extract_and_validate_ipynb
from core.ast_engine     import detect_and_evaluate_steps
from core.scoring_engine import calculate_scores, build_score_summary
from core.ollama_client  import get_feedback


def _get_dataset_schema(dataset_path: str | None) -> str | None:
    """Read column names + first 5 rows from teacher's CSV."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows   = [next(reader) for _ in range(6)]
        columns = rows[0]
        preview = rows[1:]
        lines   = [f"Columns ({len(columns)}): {', '.join(columns)}",
                   "Sample rows:"]
        for row in preview:
            lines.append("  " + ", ".join(row))
        return "\n".join(lines)
    except Exception:
        return None


def run_grader(
    notebook_path: str,
    rubric: dict,
    task_type: str,
    dataset_path: str | None = None,
    teacher_custom_subtasks: dict = None,
) -> dict:
    """
    Full grading pipeline for one submission.

    Args:
        notebook_path           : path to student .ipynb file
        rubric                  : { step: { points, depends_on } }
        task_type               : e.g. "Classification"
        dataset_path            : optional path to teacher CSV
        teacher_custom_subtasks : { step: ["custom requirement", ...] }

    Returns full result dict for DB storage.
    """
    # ── 1. Parse notebook ──────────────────────────────────
    parsed = extract_and_validate_ipynb(notebook_path)
    if parsed["status"] == "error":
        return {"status": "error", "error_message": parsed["message"]}

    code = parsed["code"]

    # ── 2. Enhanced AST analysis ───────────────────────────
    step_results, cross_violations = detect_and_evaluate_steps(
        code                    = code,
        rubric                  = rubric,
        teacher_custom_subtasks = teacher_custom_subtasks or {},
    )

    # ── 3. Proportional scoring ────────────────────────────
    scores, penalties, missed, total = calculate_scores(
        rubric         = rubric,
        step_results   = step_results,
        cross_violations = cross_violations,
    )
    max_score = sum(v["points"] for v in rubric.values())

    # Build timeline for display (ordered detected steps)
    timeline = sorted(
        [s for s, r in step_results.items() if r.detected],
        key=lambda s: step_results[s].first_line,
    )

    # ── 4. LLM feedback ────────────────────────────────────
    summary        = build_score_summary(
        rubric       = rubric,
        step_results = step_results,
        scores       = scores,
        penalties    = penalties,
        missed       = missed,
        total        = total,
    )
    dataset_schema = _get_dataset_schema(dataset_path)
    feedback       = get_feedback(
        summary        = summary,
        scores         = scores,
        missed         = missed,
        penalties      = penalties,
        rubric         = rubric,
        task_type      = task_type,
        dataset_schema = dataset_schema,
        step_results   = step_results,
    )

    # Build per-step detail for UI display
    step_detail = {}
    for step_name, result in step_results.items():
        step_detail[step_name] = {
            "completed_subtasks": result.completed_subtasks,
            "missed_subtasks":    result.missed_subtasks,
            "order_violations":   result.order_violations,
            "partial_ratio":      result.partial_ratio,
        }

    return {
        "status":             "success",
        "task_type":           task_type,
        "extracted_timeline":  timeline,
        "step_detail":         step_detail,
        "final_scores":        scores,
        "system_penalties":    penalties,
        "missed_steps":        missed,
        "final_total_score":   total,
        "max_score":           max_score,
        "feedback":            feedback,
        "graded_at":           datetime.utcnow().isoformat(),
    }
