import json
import re
import requests
from config import Config


def _build_facts(
    scores: dict,
    missed: list,
    penalties: list,
    rubric: dict,
    step_results: dict,
) -> dict:
    """
    Python pre-processes all grading data into clean categorised facts.
    The LLM receives ready-made sentences, not raw data to reason about.
    This is the key speed optimisation — reasoning in Python, writing in LLM.
    """
    facts = {
        "strengths":   [],   # full-credit steps, ready as sentences
        "weaknesses":  [],   # categorised weakness sentences
        "fixes":       [],   # what code to show per weakness
    }

    # ── Strengths — full credit steps ─────────────────────
    for step, sc in scores.items():
        if rubric.get(step) and sc >= rubric[step]["points"] and step not in missed:
            # Make a human-readable sentence in Python
            label = step.replace("_", " ")
            facts["strengths"].append(
                f"The {label} step was correctly and fully completed."
            )

    # ── Weaknesses — ordered: critical → major → partial → missing ──

    # 1. Data leakage (critical)
    for p in penalties:
        if p.get("type") == "DATA_LEAKAGE":
            step  = p["step"]
            label = step.replace("_", " ")
            facts["weaknesses"].append(
                f"🔴 The {label} step has a critical data leakage error. "
                f"The scaler was fitted on the full dataset before the "
                f"train-test split, which allows test data to influence "
                f"training and produces misleadingly optimistic results."
            )
            facts["fixes"].append(
                f"{label}: Split the data first, then fit the scaler "
                f"only on X_train and transform X_test separately.\n"
                f"```python\n"
                f"from sklearn.preprocessing import StandardScaler\n"
                f"scaler = StandardScaler()\n"
                f"X_train_scaled = scaler.fit_transform(X_train)\n"
                f"X_test_scaled  = scaler.transform(X_test)\n"
                f"```"
            )

    # 2. Cross-step order violations (major)
    for p in penalties:
        if p.get("type") == "CROSS_STEP_ORDER":
            step  = p["step"]
            label = step.replace("_", " ")
            facts["weaknesses"].append(
                f"🟡 The {label} step was performed in the wrong order. "
                f"{p['message'].split('SEQUENCE VIOLATION:')[-1].split('— Penalty')[0].strip()}"
            )
            facts["fixes"].append(
                f"{label}: Reorder your pipeline so this step comes "
                f"after its dependencies."
            )

    # 3. Entirely missing steps
    for step in missed:
        label = step.replace("_", " ")
        why = {
            "data_loading":           "Without loading data, no ML pipeline can run.",
            "eda_and_visualization":  "EDA helps understand data distribution and spot issues before modelling.",
            "basic_cleaning":         "Uncleaned data leads to biased or broken models.",
            "split_data":             "Without splitting, model performance cannot be honestly evaluated.",
            "scaling_and_imputation": "Feature scaling is important for algorithms sensitive to feature magnitude.",
            "model_training":         "No model was trained — the core ML step is missing.",
            "hyperparameter_tuning":  "Tuning improves model performance beyond default parameters.",
            "evaluation":             "Without evaluation metrics, model quality cannot be assessed.",
        }.get(step, "This step is required for a complete ML pipeline.")

        facts["weaknesses"].append(
            f"🔴 The {label} step is entirely missing. {why}"
        )
        facts["fixes"].append(_missing_step_fix(step))

    # 4. Partial steps
    for step, sc in scores.items():
        if rubric.get(step) and 0 < sc < rubric[step]["points"] and step not in missed:
            label  = step.replace("_", " ")
            result = step_results.get(step) if step_results else None
            missed_desc = []
            if result and hasattr(result, "missed_subtasks"):
                missed_desc = result.missed_subtasks

            if missed_desc:
                missing_str = "; ".join(missed_desc)
                facts["weaknesses"].append(
                    f"🟡 The {label} step is incomplete. "
                    f"The following sub-tasks were not done: {missing_str}."
                )
                facts["fixes"].append(_partial_step_fix(step, missed_desc))
            else:
                facts["weaknesses"].append(
                    f"🟡 The {label} step was only partially completed."
                )
                facts["fixes"].append(
                    f"{label}: Review the step requirements and ensure "
                    f"all required sub-tasks are included."
                )

    return facts


def _missing_step_fix(step: str) -> str:
    """Returns a ready-made code fix for a completely missing step."""
    fixes = {
        "basic_cleaning": (
            "basic cleaning: Handle missing values using fillna (not dropna — "
            "dropna removes rows and loses data) and remove duplicates.\n"
            "```python\n"
            "df.fillna(df.mean(numeric_only=True), inplace=True)\n"
            "df.drop_duplicates(inplace=True)\n"
            "```"
        ),
        "scaling_and_imputation": (
            "scaling and imputation: Fit the scaler only on X_train, "
            "then transform both sets separately.\n"
            "```python\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_train_scaled = scaler.fit_transform(X_train)\n"
            "X_test_scaled  = scaler.transform(X_test)\n"
            "```"
        ),
        "eda_and_visualization": (
            "eda and visualization: Add basic exploration before modelling.\n"
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "print(df.describe())\n"
            "df.hist(figsize=(12, 8))\n"
            "plt.show()\n"
            "```"
        ),
        "split_data": (
            "split data: Always split before any fitting.\n"
            "```python\n"
            "from sklearn.model_selection import train_test_split\n"
            "X = df.drop('target', axis=1)\n"
            "y = df['target']\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42)\n"
            "```"
        ),
        "model_training": (
            "model training: Instantiate and fit a model on training data.\n"
            "```python\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression(max_iter=1000)\n"
            "model.fit(X_train, y_train)\n"
            "```"
        ),
        "evaluation": (
            "evaluation: Generate predictions and compute metrics.\n"
            "```python\n"
            "from sklearn.metrics import accuracy_score, classification_report\n"
            "y_pred = model.predict(X_test)\n"
            "print(accuracy_score(y_test, y_pred))\n"
            "print(classification_report(y_test, y_pred))\n"
            "```"
        ),
        "hyperparameter_tuning": (
            "hyperparameter tuning: Use GridSearchCV to find best parameters.\n"
            "```python\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "param_grid = {'C': [0.1, 1, 10]}\n"
            "grid = GridSearchCV(model, param_grid, cv=5)\n"
            "grid.fit(X_train, y_train)\n"
            "print(grid.best_params_)\n"
            "```"
        ),
    }
    return fixes.get(
        step,
        f"{step.replace('_', ' ')}: Add this step to your pipeline."
    )


def _partial_step_fix(step: str, missed_desc: list) -> str:
    """Returns targeted fix code for missing sub-tasks within a partial step."""
    fix_map = {
        "Missing values handled (dropna, fillna, or SimpleImputer)": (
            "```python\n"
            "# Use fillna to impute — do NOT use dropna (it loses data)\n"
            "df.fillna(df.mean(numeric_only=True), inplace=True)\n"
            "```"
        ),
        "Duplicates removed (drop_duplicates)": (
            "```python\n"
            "df.drop_duplicates(inplace=True)\n"
            "```"
        ),
        "Predictions generated with .predict()": (
            "```python\n"
            "y_pred = model.predict(X_test)\n"
            "```"
        ),
        "At least one evaluation metric used": (
            "```python\n"
            "from sklearn.metrics import accuracy_score, classification_report\n"
            "print(accuracy_score(y_test, y_pred))\n"
            "print(classification_report(y_test, y_pred))\n"
            "```"
        ),
        "Parameter grid defined": (
            "```python\n"
            "param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 500]}\n"
            "```"
        ),
    }

    code_blocks = []
    for desc in missed_desc:
        # Match by checking if any key is contained in the description
        for key, fix in fix_map.items():
            if any(word in desc for word in key.split()[:3]):
                code_blocks.append(fix)
                break

    label = step.replace("_", " ")
    if code_blocks:
        return f"{label}: Add the missing sub-tasks:\n" + "\n".join(code_blocks)
    return f"{label}: Complete all required sub-tasks for this step."


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
    Fast single Ollama call.

    Python pre-builds all facts → LLM only writes, does not reason.
    Prompt is under 400 tokens → ~2x faster than previous version.
    """
    facts = _build_facts(scores, missed, penalties, rubric, step_results)

    # If everything is already determined by Python, just return directly
    # for edge case where there's nothing to write
    if not facts["weaknesses"] and not facts["strengths"]:
        return {
            "strengths":    ["All steps were completed correctly."],
            "weaknesses":   [],
            "improvements": [],
        }

    # Build a tight prompt — LLM just rewrites pre-built sentences naturally
    strengths_raw  = "\n".join(f"- {s}" for s in facts["strengths"])  or "- None"
    weaknesses_raw = "\n".join(f"- {w}" for w in facts["weaknesses"]) or "- None"
    fixes_raw      = "\n".join(f"- {f}" for f in facts["fixes"])      or "- None"

    prompt = f"""You are grading a {task_type} ML assignment. 
Rewrite the following pre-determined grading facts as clean, professional feedback JSON.
Do NOT change the meaning. Do NOT add or remove issues. Just improve the language quality.

STRENGTHS (rewrite each as one clean sentence):
{strengths_raw}

WEAKNESSES (keep severity emoji 🔴/🟡, rewrite each as one clear sentence):
{weaknesses_raw}

IMPROVEMENTS (keep code blocks exactly as-is, only improve the explanation text):
{fixes_raw}

Output ONLY this JSON, no markdown fences:
{{
  "strengths":    ["..."],
  "weaknesses":   ["..."],
  "improvements": ["..."]
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
                            "You are a professional ML teaching assistant. "
                            "Output only valid JSON. "
                            "Never change the technical meaning of feedback. "
                            "Keep all code blocks exactly as provided."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream":  False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500,   # down from 1200 — feedback is ~300 tokens max
                },
            },
            timeout=120,
        )
        raw   = resp.json()["message"]["content"].strip()
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            result = json.loads(clean)
            # Validate structure
            if all(k in result for k in ("strengths", "weaknesses", "improvements")):
                return result
            raise ValueError("Missing keys in response")
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
            # Fallback — return Python-built facts directly without LLM polish
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
        # On any error, return Python-built facts — never fail silently
        return {
            "strengths":    facts["strengths"],
            "weaknesses":   facts["weaknesses"] or [f"Feedback generation error: {e}"],
            "improvements": facts["fixes"],
        }
