import json
import re
import requests
from config import Config


def _build_facts(scores, missed, penalties, rubric, step_results):
    """
    Build feedback facts from grading results.
    Fully dynamic — works with any step names.
    No hardcoded templates for code suggestions.
    """
    facts = {
        "strengths":  [],
        "weaknesses": [],
        "fixes":      [],      # text only — LLM will generate code
        "fix_context": [],     # context for LLM to generate good code
    }

    # Strengths — full credit steps
    for step, sc in scores.items():
        if rubric.get(step) and sc >= rubric[step]["points"] and step not in missed:
            facts["strengths"].append(
                f"The '{step}' step was correctly and fully completed."
            )

    # Weaknesses ordered: critical → major → missing → partial
    for p in penalties:
        if p.get("type") == "DATA_LEAKAGE":
            step  = p["step"]
            facts["weaknesses"].append(
                f"🔴 The '{step}' step has a critical data leakage error. "
                f"The scaler was fitted on the full dataset before splitting, "
                f"which invalidates model evaluation results."
            )
            facts["fixes"].append(
                f"Fix for '{step}': Split the data first, then fit the scaler "
                f"only on X_train and use transform() on X_test."
            )
            facts["fix_context"].append({
                "step": step,
                "issue": "data_leakage",
                "instruction": "Show correct code: split first, then fit_transform on X_train only, transform on X_test."
            })

    for p in penalties:
        if p.get("type") == "CROSS_STEP_ORDER":
            step = p["step"]
            facts["weaknesses"].append(
                f"🟡 The '{step}' step was performed in the wrong order. "
                f"{p['message'].split('SEQUENCE VIOLATION:')[-1].split('— Penalty')[0].strip()}"
            )
            facts["fixes"].append(
                f"Fix for '{step}': Reorder this step so it follows its dependencies."
            )
            facts["fix_context"].append({
                "step": step,
                "issue": "wrong_order",
                "instruction": f"Show a brief code comment explaining the correct pipeline order for {step}."
            })

    for step in missed:
        label = step.replace("_", " ")
        facts["weaknesses"].append(
            f"🔴 The '{label}' step is entirely missing from the submission."
        )
        facts["fixes"].append(
            f"Fix for '{label}': Add this step to the ML pipeline."
        )
        facts["fix_context"].append({
            "step": step,
            "issue": "missing",
            "instruction": f"Provide a minimal working Python code example for '{step}' in a {'{task_type}'} ML pipeline."
        })

    for step, sc in scores.items():
        if rubric.get(step) and 0 < sc < rubric[step]["points"] and step not in missed:
            result = step_results.get(step) if step_results else None
            missed_desc = []
            if result and hasattr(result, "missed_subtasks"):
                missed_desc = result.missed_subtasks

            label = step.replace("_", " ")
            if missed_desc:
                facts["weaknesses"].append(
                    f"🟡 The '{label}' step is incomplete. "
                    f"Missing: {'; '.join(missed_desc)}."
                )
                facts["fixes"].append(
                    f"Fix for '{label}': Add the missing parts: {'; '.join(missed_desc)}."
                )
                facts["fix_context"].append({
                    "step": step,
                    "issue": "partial",
                    "missing": missed_desc,
                    "instruction": f"Provide code only for the missing parts: {'; '.join(missed_desc)}."
                })

    return facts


def get_feedback(
    summary: str,
    scores: dict,
    missed: list,
    penalties: list,
    rubric: dict,
    task_type: str,
    dataset_schema: str | None = None,
    step_results: dict = None,
) -> dict:
    """
    Single Ollama call. Fully dynamic — no hardcoded step logic.
    Python builds the facts, LLM writes the language and generates code.
    """
    facts = _build_facts(scores, missed, penalties, rubric, step_results)

    if not facts["weaknesses"] and not facts["strengths"]:
        return {
            "strengths":    ["All steps were completed correctly."],
            "weaknesses":   [],
            "improvements": [],
        }

    # Replace {task_type} placeholder in fix_context
    for ctx in facts["fix_context"]:
        if "instruction" in ctx:
            ctx["instruction"] = ctx["instruction"].replace("{task_type}", task_type)

    strengths_text  = "\n".join(f"- {s}" for s in facts["strengths"])  or "- None"
    weaknesses_text = "\n".join(f"- {w}" for w in facts["weaknesses"]) or "- None"
    fixes_text      = "\n".join(f"- {f}" for f in facts["fixes"])      or "- None"

    # Build code generation instructions for each fix
    code_instructions = ""
    if facts["fix_context"]:
        code_instructions = "\n\nFor IMPROVEMENTS, generate actual Python code for each fix:\n"
        for ctx in facts["fix_context"]:
            code_instructions += f"  - {ctx['step']}: {ctx['instruction']}\n"

    dataset_block = ""
    if dataset_schema:
        dataset_block = f"\n--- DATASET SCHEMA ---\n{dataset_schema}\n---\n"

    prompt = f"""You are grading a {task_type} machine learning assignment.
{dataset_block}
Rewrite these grading facts as professional feedback JSON.
Generate Python code examples for each improvement.

STRENGTHS (rewrite professionally):
{strengths_text}

WEAKNESSES (keep 🔴/🟡 tags, rewrite clearly):
{weaknesses_text}

FIXES TO IMPLEMENT:
{fixes_text}
{code_instructions}

RULES:
- Never say a tool "is used" if the step is listed as entirely missing
- For DATA LEAKAGE: always show fit_transform on X_train only, transform on X_test
- For MISSING steps: show minimal working code for a {task_type} pipeline
- For PARTIAL steps: show only the missing parts, not the full step
- Use fillna(df.mean()) not dropna for missing values
- Keep code blocks clean and runnable

Output ONLY this JSON (no markdown around it):
{{
  "strengths":    ["one sentence per item"],
  "weaknesses":   ["🔴/🟡 tagged sentence per issue"],
  "improvements": ["step_name: explanation\\n```python\\ncode\\n```"]
}}"""

    try:
        resp = requests.post(
            Config.OLLAMA_URL,
            json={
                "model":   Config.OLLAMA_MODEL,
                "messages": [
                    {
                        "role":    "system",
                        "content": (
                            "You are a strict ML grading assistant. "
                            "Output only valid JSON. "
                            "Never say a tool is used if it is listed as missing. "
                            "Always fit_transform on X_train only. "
                            "Use fillna not dropna."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream":  False,
                "options": {"temperature": 0.1, "num_predict": 600},
            },
            timeout=180,
        )
        raw   = resp.json()["message"]["content"].strip()
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            result = json.loads(clean)
            if all(k in result for k in ("strengths", "weaknesses", "improvements")):
                return result
            raise ValueError("Missing keys")
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {
                "strengths":    facts["strengths"],
                "weaknesses":   facts["weaknesses"],
                "improvements": facts["fixes"],
            }

    except requests.exceptions.ConnectionError:
        return {
            "strengths":    facts["strengths"],
            "weaknesses":   ["Could not connect to Ollama. Make sure it is running."],
            "improvements": ["Run: ollama serve"],
        }
    except Exception as e:
        return {
            "strengths":    facts["strengths"],
            "weaknesses":   facts["weaknesses"] or [f"Feedback error: {e}"],
            "improvements": facts["fixes"],
        }
