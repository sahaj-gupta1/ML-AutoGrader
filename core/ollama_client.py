import json
import re
import requests
from config import Config


# ─────────────────────────────────────────
# SAFE TEXT GENERATOR (FINAL VERSION)
# ─────────────────────────────────────────

def _generate_text(prompt: str, max_tokens: int = 1024) -> str:
    for attempt in range(2):  # 🔥 retry mechanism
        try:
            resp = requests.post(
                Config.OLLAMA_URL,
                json={
                    "model": Config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an expert ML grading assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False
                },
                timeout=120,
            )

            raw = resp.text

            # 🔍 DEBUG
            print("🔍 RAW LLM RESPONSE:", raw[:200])

            decoder = json.JSONDecoder()
            idx = 0

            while idx < len(raw):
                try:
                    data, end = decoder.raw_decode(raw[idx:])

                    if isinstance(data, dict) and "message" in data:
                        msg = data["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            return msg["content"].strip()

                    idx += end if end > 0 else 1

                except:
                    idx += 1

        except Exception as e:
            print(f"⚠️ LLM ERROR (attempt {attempt+1}):", e)

    return None  # 🔥 IMPORTANT (NOT "")


# ─────────────────────────────────────────
# SAFE JSON GENERATOR (FINAL)
# ─────────────────────────────────────────

def _generate_json(prompt: str, max_tokens: int = 800) -> dict:
    raw = _generate_text(prompt, max_tokens)

    if not raw:
        print("⚠️ LLM EMPTY RESPONSE")
        return {"execution_timeline": [], "error": "empty_response"}

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except:
        pass

    # 🔥 fallback regex extraction
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    print("⚠️ JSON PARSE FAILED")
    return {"execution_timeline": [], "error": "parse_failed"}


# ─────────────────────────────────────────
# LAYER 2
# ─────────────────────────────────────────

def annotate_code(raw_code: str, fact_sheet: dict) -> str:
    prompt = f"""Analyze the following Python code.
Add '# STEP:' comments above major blocks.

Fact Sheet: {json.dumps(fact_sheet)}

Code:
{raw_code}
"""
    annotated = _generate_text(prompt, max_tokens=2048)

    if not annotated:
        print("⚠️ Annotation failed → using raw code")
        return raw_code

    return annotated


# ─────────────────────────────────────────
# LAYER 3 (TIMELINE)
# ─────────────────────────────────────────

def extract_timeline(annotated_code: str, rubric_keys: list, task_type: str = "") -> dict:

    prompt = f"""
Map steps to categories:
{rubric_keys}

Output:
{{"execution_timeline": ["step1","step2"]}}

Code:
{annotated_code[:6000]}
"""

    result = _generate_json(prompt)

    if "execution_timeline" not in result or not isinstance(result["execution_timeline"], list):
        print("❌ DEBUG L3 FAIL:", result)
        return {"execution_timeline": []}

    result["execution_timeline"] = [
        step for step in result["execution_timeline"]
        if step in rubric_keys
    ]

    print("✅ Timeline:", result["execution_timeline"])
    return result


# ─────────────────────────────────────────
# LAYER 4 (AUDIT)
# ─────────────────────────────────────────

def audit_timeline(raw_code: str, proposed_timeline: list, rubric_keys: list) -> dict:
    if not isinstance(proposed_timeline, list) or not proposed_timeline:
        return {"execution_timeline": []}

    prompt = f"""
Verify timeline:
{proposed_timeline}

Code:
{raw_code[:4000]}

Return JSON.
"""

    result = _generate_json(prompt)

    if "execution_timeline" in result:
        result["execution_timeline"] = [
            step for step in result["execution_timeline"]
            if step in rubric_keys
        ]
        return result

    return {"execution_timeline": proposed_timeline}


# ─────────────────────────────────────────
# LAYER 6 (FEEDBACK)
# ─────────────────────────────────────────

def get_feedback(summary, scores, missed, penalties, rubric, task_type, dataset_schema=None, step_results=None):

    if not scores:
        return {
            "strengths": [],
            "weaknesses": ["Could not evaluate submission properly"],
            "improvements": []
        }

    prompt = f"""
Give feedback for ML assignment.

Scores: {scores}
Missed: {missed}

Return JSON:
{{
 "strengths": [],
 "weaknesses": [],
 "improvements": []
}}
"""

    raw = _generate_text(prompt)

    if not raw:
        print("⚠️ Feedback generation failed")
        return {
            "strengths": [],
            "weaknesses": ["Feedback generation failed"],
            "improvements": []
        }

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except:
        print("⚠️ Feedback JSON parse failed")
        return {
            "strengths": [],
            "weaknesses": ["Invalid feedback format"],
            "improvements": []
        }