"""
Dynamic Scoring Engine
======================
Works with any rubric — known or unknown step names.

Known steps  → hybrid proportional scoring with quality multipliers
Dynamic steps → binary: detected = full marks, not detected = zero
Data leakage → max_pts * 0.15 regardless of step name
"""

from core.ast_engine import StepResult

QUALITY_MULTIPLIERS = {
    "data_leakage": 0.15,
}


def _round_to_half(value: float) -> float:
    return round(value * 2) / 2


def calculate_scores(
    rubric: dict,
    step_results: dict,
    cross_violations: list,
) -> tuple:
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

        method = result.detection_method  # 'known' or 'dynamic'

        # ── Known steps: full hybrid scoring ─────────────────
        if method == "known":
            norm = step_name.lower().replace(" ","_").replace("&","and")

            # Special: scaling leakage check
            if norm == "scaling_and_imputation":
                scaler_present = "scaler_present"   in result.completed_subtasks
                fit_on_train   = "fit_on_train_only" in result.completed_subtasks

                if not scaler_present:
                    missed.append(step_name)
                    scores[step_name] = 0.0
                    continue
                elif scaler_present and not fit_on_train:
                    score = _round_to_half(max_pts * QUALITY_MULTIPLIERS["data_leakage"])
                    scores[step_name] = score
                    penalties.append({
                        "type":     "DATA_LEAKAGE",
                        "step":     step_name,
                        "penalty":  _round_to_half(max_pts - score),
                        "severity": "critical",
                        "message":  (
                            f"DATA LEAKAGE in '{step_name}': scaler was fitted on the full "
                            f"dataset before train-test split. fit_transform() must be called "
                            f"only on X_train, then transform() on X_test. "
                            f"Penalty: -{_round_to_half(max_pts - score)} pts"
                        ),
                    })
                    continue
                else:
                    scores[step_name] = max_pts
                    continue

            # General proportional scoring
            total = len(result.completed_subtasks) + len(result.missed_subtasks)
            ratio = len(result.completed_subtasks) / total if total > 0 else 1.0
            base_score = _round_to_half(ratio * max_pts)

            # Info notes for missed sub-tasks
            for desc in result.missed_subtasks:
                penalties.append({
                    "type":     "INCOMPLETE_STEP",
                    "step":     step_name,
                    "penalty":  0,
                    "severity": "info",
                    "message":  f"'{step_name}' incomplete — {desc}",
                })

            scores[step_name] = base_score

        # ── Dynamic steps: binary scoring ─────────────────────
        else:
            # Detected = full marks, not detected = 0 (already handled above)
            scores[step_name] = max_pts

    # ── Cross-step ordering violations ────────────────────────
    applied = set()
    for msg in cross_violations:
        penalised_step = None
        for step_name in rubric:
            if f"'{step_name}'" in msg or step_name.lower() in msg.lower():
                penalised_step = step_name
                break
        if penalised_step and penalised_step not in applied:
            max_pts   = float(rubric[penalised_step]["points"])
            deduction = _round_to_half(max_pts * 0.5)
            min_score = _round_to_half(max_pts * 0.25) if scores[penalised_step] > 0 else 0.0
            scores[penalised_step] = max(min_score, scores[penalised_step] - deduction)
            applied.add(penalised_step)
            penalties.append({
                "type":     "CROSS_STEP_ORDER",
                "step":     penalised_step,
                "penalty":  deduction,
                "severity": "major",
                "message":  f"SEQUENCE VIOLATION: {msg} — Penalty: -{deduction} pts",
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
    max_score = sum(v["points"] for v in rubric.values())
    lines = [f"Total Score: {total:.1f} / {max_score}", ""]

    lines.append("Step-by-step results:")
    for step_name, info in rubric.items():
        pts     = scores.get(step_name, 0)
        max_pts = info["points"]
        result  = step_results.get(step_name)

        if step_name in missed:
            label = "ENTIRELY MISSING"
        elif pts == max_pts:
            label = "FULL MARKS"
        elif pts == 0:
            label = "ZERO — critical error"
        else:
            pct   = int((pts / max_pts) * 100)
            label = f"PARTIAL — {pct}% completed"

        lines.append(f"  {step_name:<35} {pts:.1f} / {max_pts}  [{label}]")

        if result and step_name not in missed:
            if result.detection_method == "known":
                if result.completed_subtasks:
                    lines.append(f"    STUDENT DID    : {', '.join(result.completed_subtasks)}")
                if result.missed_subtasks:
                    lines.append(f"    STUDENT MISSED : {'; '.join(result.missed_subtasks)}")
            else:
                if result.completed_subtasks:
                    lines.append(f"    EVIDENCE FOUND : {', '.join(str(e) for e in result.completed_subtasks[:5])}")

    critical = [p for p in penalties if p.get("severity") == "critical"]
    major    = [p for p in penalties if p.get("severity") == "major"]

    if critical:
        lines.append("")
        lines.append("CRITICAL ERRORS:")
        for p in critical:
            lines.append(f"  {p['message']}")

    if major:
        lines.append("")
        lines.append("MAJOR ERRORS:")
        for p in major:
            lines.append(f"  {p['message']}")

    if missed:
        lines.append("")
        lines.append("Steps entirely missing:")
        for s in missed:
            lines.append(f"  {s}  (0 / {rubric[s]['points']} pts)")

    return "\n".join(lines)
