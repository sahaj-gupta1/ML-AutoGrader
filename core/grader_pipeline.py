"""
Grader Pipeline  —  6-Layer AutoGrader Engine
=============================================
Replaces the old "Code → LLM → Feedback" with a structured pipeline:

  Layer 1 | AST Fact Extractor    | ast_engine.extract_fact_sheet()
  Layer 2 | Semantic Annotator    | ollama: adds # STEP: comments
  Layer 3 | Timeline Extractor    | ollama: maps steps → rubric categories
  Layer 4 | Audit / Hallucination | ollama: validates the timeline
  Layer 5 | Scoring Engine        | pure Python: penalties + dependency check
  Layer 6 | Feedback Generator    | ollama: human-readable report

Architecture from: samaeas20.ipynb (Qwen2.5 AutoGrader Notebook)

run_grader() signature is unchanged so app.py / grading_queue.py
do NOT need to be modified.
"""

import csv
import os
from datetime import datetime

from core.ipynb_parser  import extract_and_validate_ipynb
from core.ast_engine    import extract_fact_sheet, fact_sheet_to_dict, calculate_final_scores
from core.ollama_client import (
    annotate_code,       # Layer 2
    extract_timeline,    # Layer 3
    audit_timeline,      # Layer 4
    get_feedback,        # Layer 6
)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

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
        lines   = [f"Columns ({len(columns)}): {', '.join(columns)}", "Sample rows:"]
        for row in preview:
            lines.append("  " + ", ".join(row))
        return "\n".join(lines)
    except Exception:
        return None


def _build_step_detail(rubric: dict, timeline: list, scores: dict, penalties: list) -> dict:
    """Build per-step breakdown dict for UI display."""
    penalty_by_step = {}
    for p in penalties:
        # Extract step name from penalty message
        for step in rubric:
            if f"'{step}'" in p:
                penalty_by_step.setdefault(step, []).append(p)
                break

    detail = {}
    for step in rubric:
        detail[step] = {
            "detected":        step in timeline,
            "score":           scores.get(step, 0.0),
            "max_points":      float(rubric[step]["points"]),
            "order_violations": penalty_by_step.get(step, []),
            "partial_ratio":   (
                scores.get(step, 0.0) / float(rubric[step]["points"])
                if rubric[step]["points"] > 0 else 0.0
            ),
        }
    return detail


# ─────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_grader(
    notebook_path:           str,
    rubric:                  dict,
    task_type:               str,
    dataset_path:            str | None = None,
    teacher_custom_subtasks: dict       = None,
) -> dict:
    """
    Full 6-layer grading pipeline for one .ipynb submission.

    Args:
        notebook_path           : path to student .ipynb file
        rubric                  : { step: { "points": int, "depends_on": [...] } }
        task_type               : e.g. "Classification", "Regression", "Deep Learning"
        dataset_path            : optional path to teacher CSV (for schema hint)
        teacher_custom_subtasks : legacy arg, currently unused (kept for compatibility)

    Returns:
        Full result dict stored in Submission.result_json
    """

    print(f"\n[Pipeline] Starting grading: {os.path.basename(notebook_path)}")
    rubric_keys = list(rubric.keys())
    max_score   = float(sum(v["points"] for v in rubric.values()))

    # ──────────────────────────────────────────────────────
    # LAYER 1 — AST Fact Extractor (deterministic, cell-by-cell)
    # ──────────────────────────────────────────────────────
    print("[Layer 1] Extracting facts from notebook cells...")

    try:
        fact_sheet = extract_fact_sheet(notebook_path)
    except FileNotFoundError as e:
        return {"status": "error", "error_message": str(e)}

    if fact_sheet.parseable_cells == 0:
        return {
            "status":        "error",
            "error_message": (
                f"No parseable code cells found. "
                f"{fact_sheet.failed_cells} cell(s) had syntax errors."
            ),
        }

    facts_dict    = fact_sheet_to_dict(fact_sheet)
    combined_code = fact_sheet.combined_code

    print(
        f"[Layer 1] Done — {fact_sheet.parseable_cells} cells parsed, "
        f"{fact_sheet.failed_cells} skipped. "
        f"Imports: {len(fact_sheet.imports)}, Functions: {len(fact_sheet.functions)}"
    )

    # ──────────────────────────────────────────────────────
    # LAYER 2 — Semantic Annotator (LLM)
    # Inserts # STEP: comments above every logical block
    # ──────────────────────────────────────────────────────
    print("[Layer 2] Annotating code with semantic step labels...")

    annotated = annotate_code(combined_code, facts_dict)
    print("[Layer 2] Done.")

    # ──────────────────────────────────────────────────────
    # LAYER 3 — Timeline Extractor (LLM)
    # Maps # STEP: comments → rubric category names
    # ──────────────────────────────────────────────────────
    print("[Layer 3] Extracting execution timeline from annotations...")

    layer3_result  = extract_timeline(annotated, rubric_keys, task_type)
    draft_timeline = (
        layer3_result.get("execution_timeline", [])
        if isinstance(layer3_result, dict) else []
    )
    print(f"[Layer 3] Draft timeline: {draft_timeline}")

    # ──────────────────────────────────────────────────────
    # LAYER 4 — Audit / Anti-Hallucination (LLM)
    # Validates that every step in the timeline really exists in the code
    # ──────────────────────────────────────────────────────
    print("[Layer 4] Auditing timeline for hallucinations...")

    layer4_result  = audit_timeline(combined_code, draft_timeline, rubric_keys)
    final_timeline = (
        layer4_result.get("execution_timeline", draft_timeline)
        if isinstance(layer4_result, dict) else draft_timeline
    )

    # Deduplicate while preserving order
    seen = set()
    final_timeline = [s for s in final_timeline if not (s in seen or seen.add(s))]

    print(f"[Layer 4] Final timeline: {final_timeline}")

    # ──────────────────────────────────────────────────────
    # LAYER 5 — Scoring Engine (pure Python)
    # Awards base points, detects sequence violations,
    # applies dependency penalties
    # ──────────────────────────────────────────────────────
    print("[Layer 5] Calculating scores and penalties...")

    scoring = calculate_final_scores(rubric, final_timeline)
    scores        = scoring["scores"]
    penalties     = scoring["penalties"]
    missed_steps  = scoring["missed_steps"]
    total_score   = scoring["total"]

    print(f"[Layer 5] Total: {total_score}/{max_score}  |  Missed: {missed_steps}")

    # ──────────────────────────────────────────────────────
    # LAYER 6 — Feedback Generator (LLM)
    # Writes human-readable strengths / weaknesses / improvements
    # based strictly on Layer 5 data (no hallucination possible)
    # ──────────────────────────────────────────────────────
    print("[Layer 6] Generating student feedback...")

    dataset_schema = _get_dataset_schema(dataset_path)

    if not final_timeline:
        feedback = {
            "strengths":    [],
            "weaknesses":   ["System could not parse any executable steps from your notebook."],
            "improvements": ["Ensure your notebook runs without errors and follows the assignment structure."],
        }
    else:
        # Convert penalty strings to dicts for ollama_client compatibility
        penalty_dicts = []
        for p in penalties:
            if "VIOLATION" in p:
                penalty_dicts.append({"type": "CROSS_STEP_ORDER", "message": p, "step": _extract_step(p, rubric_keys)})
            else:
                penalty_dicts.append({"type": "MISSING_DEPENDENCY", "message": p, "step": _extract_step(p, rubric_keys)})

        feedback = get_feedback(
            summary        = _build_summary(scores, missed_steps, total_score, max_score),
            scores         = scores,
            missed         = missed_steps,
            penalties      = penalty_dicts,
            rubric         = rubric,
            task_type      = task_type,
            dataset_schema = dataset_schema,
            step_results   = None,
        )

    print("[Layer 6] Done.")

    # ──────────────────────────────────────────────────────
    # BUILD FINAL RESULT
    # ──────────────────────────────────────────────────────
    step_detail = _build_step_detail(rubric, final_timeline, scores, penalties)

    return {
        "status":             "success",
        "task_type":           task_type,
        "extracted_timeline":  final_timeline,
        "step_detail":         step_detail,
        "final_scores":        scores,
        "system_penalties":    penalties,
        "missed_steps":        missed_steps,
        "final_total_score":   total_score,
        "max_score":           max_score,
        "feedback":            feedback,
        "ast_facts": {
            "imports":          fact_sheet.imports,
            "functions":        fact_sheet.functions,
            "parseable_cells":  fact_sheet.parseable_cells,
            "failed_cells":     fact_sheet.failed_cells,
        },
        "graded_at":           datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _extract_step(penalty_msg: str, rubric_keys: list) -> str:
    """Pull the step name out of a penalty message string."""
    for step in rubric_keys:
        if f"'{step}'" in penalty_msg:
            return step
    return "unknown"


def _build_summary(scores: dict, missed: list, total: float, max_score: float) -> str:
    """Build a one-paragraph text summary for the LLM feedback prompt."""
    lines = [f"Score: {total}/{max_score}"]
    for step, score in scores.items():
        lines.append(f"  {step}: {score}")
    if missed:
        lines.append(f"Missed steps: {', '.join(missed)}")
    return "\n".join(lines)
