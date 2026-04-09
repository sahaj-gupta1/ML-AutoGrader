"""
Ollama Client
=============
Handles ALL LLM calls in the grading pipeline.

Layer mapping:
  Layer 2  annotate_code()      — adds # STEP: comments to code
  Layer 3  extract_timeline()   — maps steps → rubric categories
  Layer 4  audit_timeline()     — validates timeline, removes hallucinations
  Layer 6  get_feedback()       — generates student-facing feedback report

Layers 1 and 5 are pure Python (ast_engine.py) — no LLM needed.
"""

import json
import re
import requests
from config import Config


# ─────────────────────────────────────────────────────────────
#  LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────

def _generate_text(prompt: str, max_tokens: int = 1024) -> str:
    """
    Basic Ollama call — returns raw text.
    Falls back gracefully on connection errors.
    """
    try:
        resp = requests.post(
            Config.OLLAMA_URL,
            json={
                "model":    Config.OLLAMA_MODEL,
                "messages": [
                    {
                        "role":    "system",
                        "content": "You are an expert Python machine learning grading assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream":  False,
                "options": {"temperature": 0.1, "num_predict": max_tokens},
            },
            timeout=180,
        )
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return ""
    except Exception as e:
        return f"[LLM Error: {e}]"


def _generate_json(prompt: str, max_tokens: int = 800) -> dict:
    """
    Ollama call that expects JSON output.
    Tries multiple extraction strategies before giving up.
    """
    raw = _generate_text(prompt + "\nOutput ONLY valid JSON. No markdown fences.", max_tokens)

    # Strategy 1: ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: clean and parse directly
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first {...} block
    brace_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {"error": "JSON parse failed", "raw": raw[:300]}


# ─────────────────────────────────────────────────────────────
#  LAYER 2 — SEMANTIC ANNOTATOR
# ─────────────────────────────────────────────────────────────

def annotate_code(raw_code: str, fact_sheet: dict) -> str:
    """
    Layer 2: Ask the LLM to insert '# STEP: <description>' comments
    above every major logical block in the student's code.

    The fact_sheet prevents hallucination — LLM can only reference
    functions that actually appear in the AST.

    Args:
        raw_code   : combined parseable cell code from FactSheet
        fact_sheet : dict with 'imports' and 'functions' keys

    Returns:
        annotated code string (original code + # STEP: comments)
    """
    prompt = f"""Analyze the following Python code from a student's ML assignment.
Insert a descriptive comment starting with `# STEP:` above every major logical block.
The comment should explain the functional intent of that block (e.g. data loading, model training).

Rules:
- Do NOT change any Python code. Only add # STEP: comments.
- Use ONLY functions that appear in this Fact Sheet (do not hallucinate).
- Fact Sheet: {json.dumps(fact_sheet)}

Code:
{raw_code}
"""
    annotated = _generate_text(prompt, max_tokens=2048)

    # Fallback: if LLM fails or returns empty, use raw code
    if not annotated.strip() or "[LLM Error" in annotated:
        return raw_code

    return annotated


# ─────────────────────────────────────────────────────────────
#  LAYER 3 — TIMELINE EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_timeline(annotated_code: str, rubric_keys: list, task_type: str = "") -> dict:
    """
    Layer 3: Map the # STEP: comments in annotated code to exact
    rubric category names. Returns a chronological execution_timeline.

    Args:
        annotated_code : output of annotate_code()
        rubric_keys    : list of rubric step names (e.g. ["data_loading", ...])
        task_type      : "Classification", "Regression", "Deep Learning", etc.

    Returns:
        {"execution_timeline": ["step1", "step2", ...]}
    """
    task_hint = ""
    if task_type:
        task_hint = f"\nThis is a {task_type} assignment."

    prompt = f"""Map the `# STEP:` comments in the code below to these EXACT rubric categories:
{rubric_keys}
{task_hint}

UNIVERSAL BEHAVIORAL RULES:
- data_loading        : pd.read_csv, pd.read_excel, or any file read
- basic_cleaning      : .dropna(), .fillna(), LabelEncoder, get_dummies, drop_duplicates
- split_data          : train_test_split specifically
- scaling_and_imputation : StandardScaler, MinMaxScaler, SimpleImputer
- eda_and_visualization  : .describe(), .info(), plt.plot, sns.heatmap, .hist()

MACHINE LEARNING (SVM, Trees, Forests, Regression):
- model_training      : .fit() on sklearn models (SVC, RandomForest, LogisticRegression, etc.)
- hyperparameter_tuning : GridSearchCV or RandomizedSearchCV
- evaluation          : r2_score, accuracy_score, mean_squared_error, classification_report

DEEP LEARNING (Keras / PyTorch):
- model_architecture  : Sequential(), add(Dense(...)), nn.Module definitions
- model_compilation   : model.compile(optimizer=..., loss=...)
- model_training      : model.fit(..., epochs=...)
- evaluation          : model.evaluate() or standard metrics

RULES:
1. Only output categories that are in: {rubric_keys}
2. Preserve chronological order as seen in the code
3. Do NOT repeat a category

Output JSON:
{{"execution_timeline": ["cat_1", "cat_2", "..."]}}

Code:
{annotated_code[:6000]}
"""
    result = _generate_json(prompt)

    # Validate — keep only categories that exist in rubric_keys
    if isinstance(result, dict) and "execution_timeline" in result:
        result["execution_timeline"] = [
            step for step in result["execution_timeline"]
            if step in rubric_keys
        ]

    return result


# ─────────────────────────────────────────────────────────────
#  LAYER 4 — AUDIT / HALLUCINATION CHECK
# ─────────────────────────────────────────────────────────────

def audit_timeline(raw_code: str, proposed_timeline: list, rubric_keys: list) -> dict:
    """
    Layer 4: Ask the LLM to verify the proposed timeline against the
    raw code. Removes hallucinated steps (steps claimed but not present).

    Args:
        raw_code          : original combined code (not annotated)
        proposed_timeline : output of extract_timeline()
        rubric_keys       : list of valid rubric step names

    Returns:
        {"execution_timeline": [...]} — corrected timeline
    """
    if not proposed_timeline:
        return {"execution_timeline": []}

    prompt = f"""You are a grading auditor. Verify the proposed timeline against the raw code.

Raw Code (first 4000 chars):
{raw_code[:4000]}

Proposed Timeline: {proposed_timeline}
Valid Categories:  {rubric_keys}

AUDIT RULES:
1. Remove any step that has NO evidence in the code (hallucination).
2. If a step IS present but was missed, add it.
3. Preserve the correct chronological order.
4. Only include categories from: {rubric_keys}

If the proposed timeline is already accurate, return it unchanged.

Output JSON:
{{"execution_timeline": ["cat_1", "cat_2", "..."]}}
"""
    result = _generate_json(prompt)

    # Validate output — keep only valid rubric keys
    if isinstance(result, dict) and "execution_timeline" in result:
        result["execution_timeline"] = [
            step for step in result["execution_timeline"]
            if step in rubric_keys
        ]
        return result

    # If LLM fails, trust the draft timeline
    return {"execution_timeline": proposed_timeline}


# ─────────────────────────────────────────────────────────────
#  LAYER 6 — FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────

def _build_facts(
    scores: dict,
    missed: list,
    penalties: list,
    rubric: dict,
    step_results: dict,
) -> dict:
    """
    Python pre-processes all grading data into clean categorised facts.
    The LLM receives ready-made sentences — it only polishes language.
    """
    facts = {"strengths": [], "weaknesses": [], "fixes": []}

    # Strengths — full-credit steps
    for step, sc in scores.items():
        if rubric.get(step) and sc >= rubric[step]["points"] and step not in missed:
            label = step.replace("_", " ")
            facts["strengths"].append(
                f"The {label} step was correctly and fully completed."
            )

    # Weaknesses — ordered: critical → major → missing → partial
    for p in penalties:
        p_type = p.get("type", "") if isinstance(p, dict) else ""
        p_msg  = p.get("message", p) if isinstance(p, dict) else str(p)
        step   = p.get("step", "")  if isinstance(p, dict) else ""
        label  = step.replace("_", " ") if step else "step"

        if p_type == "DATA_LEAKAGE":
            facts["weaknesses"].append(
                f"🔴 {label}: Critical data leakage — scaler fitted before train/test split."
            )
            facts["fixes"].append(
                f"{label}: Fit scaler only on X_train.\n"
                f"```python\nscaler = StandardScaler()\n"
                f"X_train_s = scaler.fit_transform(X_train)\n"
                f"X_test_s  = scaler.transform(X_test)\n```"
            )
        elif "VIOLATION" in p_msg or p_type == "CROSS_STEP_ORDER":
            facts["weaknesses"].append(
                f"🟡 {label}: Executed in wrong order — {p_msg.split('Penalty')[0].strip()}"
            )
            facts["fixes"].append(
                f"{label}: Reorder your pipeline so this step follows its dependencies."
            )
        elif "MISSING DEPENDENCY" in p_msg:
            facts["weaknesses"].append(
                f"🟡 {label}: {p_msg}"
            )

    for step in missed:
        label = step.replace("_", " ")
        facts["weaknesses"].append(
            f"🔴 {label}: This step is entirely missing from the submission."
        )
        facts["fixes"].append(_missing_step_fix(step))

    for step, sc in scores.items():
        if rubric.get(step) and 0 < sc < rubric[step]["points"] and step not in missed:
            label = step.replace("_", " ")
            facts["weaknesses"].append(
                f"🟡 {label}: Only partially completed ({sc}/{rubric[step]['points']} pts)."
            )

    return facts


def _missing_step_fix(step: str) -> str:
    fixes = {
        "basic_cleaning": (
            "basic cleaning: Handle missing values and remove duplicates.\n"
            "```python\ndf.fillna(df.mean(numeric_only=True), inplace=True)\n"
            "df.drop_duplicates(inplace=True)\n```"
        ),
        "scaling_and_imputation": (
            "scaling: Fit scaler only on training data.\n"
            "```python\nscaler = StandardScaler()\n"
            "X_train_s = scaler.fit_transform(X_train)\n"
            "X_test_s  = scaler.transform(X_test)\n```"
        ),
        "split_data": (
            "split data: Always split before fitting.\n"
            "```python\nfrom sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n```"
        ),
        "model_training": (
            "model training: Fit a model on training data.\n"
            "```python\nfrom sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression()\nmodel.fit(X_train, y_train)\n```"
        ),
        "evaluation": (
            "evaluation: Compute metrics on test data.\n"
            "```python\nfrom sklearn.metrics import accuracy_score\n"
            "print(accuracy_score(y_test, model.predict(X_test)))\n```"
        ),
        "data_loading": (
            "data loading: Load your dataset.\n"
            "```python\nimport pandas as pd\ndf = pd.read_csv('your_data.csv')\n```"
        ),
        "hyperparameter_tuning": (
            "hyperparameter tuning: Use GridSearchCV.\n"
            "```python\nfrom sklearn.model_selection import GridSearchCV\n"
            "grid = GridSearchCV(model, {'C': [0.1,1,10]}, cv=5)\n"
            "grid.fit(X_train, y_train)\n```"
        ),
    }
    return fixes.get(step, f"{step.replace('_', ' ')}: Add this required step.")


def get_feedback(
    summary:        str,
    scores:         dict,
    missed:         list,
    penalties:      list,
    rubric:         dict,
    task_type:      str,
    dataset_schema: str | None = None,
    step_results:   dict       = None,
) -> dict:
    """
    Layer 6: Generate student-facing feedback.

    Python pre-builds all facts first (strengths / weaknesses / fixes).
    LLM only polishes the language — it cannot add or remove issues.
    This makes feedback deterministic and hallucination-free.
    """
    facts = _build_facts(scores, missed, penalties, rubric, step_results)

    # Edge case: nothing to report
    if not facts["weaknesses"] and not facts["strengths"]:
        return {
            "strengths":    ["All steps were completed correctly."],
            "weaknesses":   [],
            "improvements": [],
        }

    strengths_raw  = "\n".join(f"- {s}" for s in facts["strengths"])  or "- None"
    weaknesses_raw = "\n".join(f"- {w}" for w in facts["weaknesses"]) or "- None"
    fixes_raw      = "\n".join(f"- {f}" for f in facts["fixes"])      or "- None"

    prompt = f"""You are grading a {task_type} ML assignment.
Rewrite the following pre-determined grading facts as clean, professional feedback JSON.
Do NOT change the meaning. Do NOT add or remove issues. Just improve language quality.

STRENGTHS (rewrite each as one clean sentence):
{strengths_raw}

WEAKNESSES (keep 🔴/🟡 emoji, rewrite each clearly):
{weaknesses_raw}

IMPROVEMENTS (keep code blocks exactly — only improve explanation text):
{fixes_raw}

Output ONLY this JSON (no markdown fences):
{{
  "strengths":    ["..."],
  "weaknesses":   ["..."],
  "improvements": ["..."]
}}"""

    try:
        resp = requests.post(
            Config.OLLAMA_URL,
            json={
                "model":    Config.OLLAMA_MODEL,
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
                "options": {"temperature": 0.1, "num_predict": 600},
            },
            timeout=120,
        )
        raw   = resp.json()["message"]["content"].strip()
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            result = json.loads(clean)
            if all(k in result for k in ("strengths", "weaknesses", "improvements")):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())

    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass

    # Fallback — return Python-built facts directly (always works)
    return {
        "strengths":    facts["strengths"],
        "weaknesses":   facts["weaknesses"],
        "improvements": facts["fixes"],
    }
