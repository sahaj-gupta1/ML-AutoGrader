"""
Hybrid Scoring Engine
=====================
score = base_completion_score * quality_multiplier

Base score  → proportional to sub-tasks completed
Multiplier  → reflects quality / correctness of implementation

Quality multipliers (relative to max_pts, dynamic rubric safe):
  perfect        1.0   all sub-tasks done correctly
  minor_issue    0.75  ordering note, style issue
  partial        N/A   uses completion ratio directly
  data_leakage   0.15  scaler present but fitted on full X
  absent         0.0   step completely missing
"""

from core.ast_engine import StepResult

# Quality multipliers — tunable without touching logic
QUALITY_MULTIPLIERS = {
    "perfect":      1.0,
    "minor_issue":  0.75,
    "data_leakage": 0.15,
    "absent":       0.0,
}


def _round_to_half(value: float) -> float:
    """Round to nearest 0.5."""
    return round(value * 2) / 2


def calculate_scores(
    rubric: dict,
    step_results: dict,
    cross_violations: list,
) -> tuple:
    """
    Returns: scores, penalties, missed, total

    All penalties included in list with severity field:
      'critical' → data leakage, shown as 🔴
      'major'    → cross-step ordering, shown as 🟡
      'info'     → incomplete sub-tasks, shown as ℹ
    """
    scores    = {step: 0.0 for step in rubric}
    penalties = []
    missed    = []

    for step_name, info in rubric.items():
        max_pts = float(info["points"])
        result  = step_results.get(step_name)

        if result is None or not result.detected:
            missed.append(step_name)
            scores[step_name] = 0.0
            continue

        # ── Special handling: scaling_and_imputation ──────
        # Needs quality check beyond simple completion ratio
        if step_name == "scaling_and_imputation":
            scaler_present = "scaler_present"   in result.completed_subtasks
            fit_on_train   = "fit_on_train_only" in result.completed_subtasks

            if not scaler_present:
                # Scaler completely absent — zero, treat as missing
                missed.append(step_name)
                scores[step_name] = 0.0
                # Remove from detected so feedback is "entirely missing"
                continue

            elif scaler_present and not fit_on_train:
                # Scaler exists but data leakage — critical error
                score = _round_to_half(
                    max_pts * QUALITY_MULTIPLIERS["data_leakage"]
                )
                scores[step_name] = score
                penalties.append({
                    "type":     "DATA_LEAKAGE",
                    "step":     step_name,
                    "penalty":  _round_to_half(max_pts - score),
                    "severity": "critical",
                    "message":  (
                        f"DATA LEAKAGE in '{step_name}': scaler was "
                        f"fitted on the full dataset before train-test "
                        f"split. fit_transform() must be called only on "
                        f"X_train, then transform() on X_test. "
                        f"This invalidates evaluation results. "
                        f"Penalty: -{_round_to_half(max_pts - score)} pts"
                    ),
                })
                continue

            else:
                # Scaler present and correctly used — full marks
                scores[step_name] = max_pts
                continue

        # ── General steps: hybrid scoring ─────────────────
        total_subtasks = len(result.completed_subtasks) + \
                         len(result.missed_subtasks)

        if total_subtasks == 0:
            ratio = 1.0 if result.detected else 0.0
        else:
            ratio = len(result.completed_subtasks) / total_subtasks

        # Base score from completion ratio
        base_score = _round_to_half(ratio * max_pts)

        # Add info notes for missed sub-tasks
        # (mark already reduced via ratio, these notes appear in UI)
        if result.missed_subtasks:
            for desc in result.missed_subtasks:
                penalties.append({
                    "type":     "INCOMPLETE_STEP",
                    "step":     step_name,
                    "penalty":  0,
                    "severity": "info",
                    "message":  f"'{step_name}' incomplete — {desc}",
                })

        # Within-step order issues — info note only, no deduction
        for msg in result.order_violations:
            penalties.append({
                "type":     "ORDER_NOTE",
                "step":     step_name,
                "penalty":  0,
                "severity": "info",
                "message":  f"Note for '{step_name}': {msg}",
            })

        scores[step_name] = base_score

    # ── Cross-step ordering violations ────────────────────
    applied = set()
    for msg in cross_violations:
        penalised_step = None
        for step_name in rubric:
            if f"'{step_name}'" in msg:
                penalised_step = step_name
                break

        if penalised_step and penalised_step not in applied:
            max_pts   = float(rubric[penalised_step]["points"])
            deduction = _round_to_half(max_pts * 0.5)
            # Never reduce below 25% of max — student did attempt the step
            min_score = _round_to_half(max_pts * 0.25) if scores[penalised_step] > 0 else 0.0
            scores[penalised_step] = max(
                min_score, scores[penalised_step] - deduction
            )
            applied.add(penalised_step)
            penalties.append({
                "type":     "CROSS_STEP_ORDER",
                "step":     penalised_step,
                "penalty":  deduction,
                "severity": "major",
                "message":  (
                    f"SEQUENCE VIOLATION: {msg} "
                    f"— Penalty: -{deduction} pts"
                ),
            })

    total = sum(scores.values())
    return scores, penalties, missed, total


def build_score_summary(
    rubric: dict,
    step_results: dict,
    scores: dict,
    penalties: list,
    missed: list,
    total: float,
) -> str:
    """
    Detailed summary for LLM prompt.
    Uses unambiguous language — completed vs missing, never mixes them.
    """
    max_score = sum(v["points"] for v in rubric.values())
    lines = [f"Total Score: {total:.1f} / {max_score}", ""]

    lines.append("Step-by-step results:")
    for step_name, info in rubric.items():
        pts     = scores.get(step_name, 0)
        max_pts = info["points"]
        result  = step_results.get(step_name)

        # Determine label
        if step_name in missed:
            label = "ENTIRELY MISSING"
        elif pts == max_pts:
            label = "FULL MARKS"
        elif pts == 0:
            label = "ZERO — critical error"
        else:
            pct   = int((pts / max_pts) * 100)
            label = f"PARTIAL — {pct}% completed"

        lines.append(f"  {step_name:<32} {pts:.1f} / {max_pts}  [{label}]")

        if result and not step_name in missed:
            # Clearly separate completed vs missing
            done = result.completed_subtasks
            miss = result.missed_subtasks

            if done:
                lines.append(
                    f"    STUDENT DID     : {', '.join(done)}"
                )
            if miss:
                # Pass descriptions, not sub-task names
                # so LLM gets human-readable text not variable names
                lines.append(
                    f"    STUDENT MISSED  : {'; '.join(miss)}"
                )

    critical = [p for p in penalties if p["severity"] == "critical"]
    major    = [p for p in penalties if p["severity"] == "major"]

    if critical:
        lines.append("")
        lines.append("CRITICAL ERRORS (heavy penalties):")
        for p in critical:
            lines.append(f"  {p['message']}")

    if major:
        lines.append("")
        lines.append("MAJOR ERRORS (sequence violations):")
        for p in major:
            lines.append(f"  {p['message']}")

    if missed:
        lines.append("")
        lines.append("Steps the student did not attempt at all:")
        for s in missed:
            lines.append(f"  {s}  (0 / {rubric[s]['points']} pts)")

    return "\n".join(lines)
