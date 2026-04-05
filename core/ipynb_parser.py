import json
import ast
import re


def _unwrap_string_code(src: str) -> str:
    """
    Handles notebooks where student code is stored inside a
    triple-quoted string variable, e.g.:

        test_code = \"\"\"
        import pandas as pd
        ...
        \"\"\"

    Extracts the content from inside the quotes so the AST
    can see the real imports and function calls.
    Falls back to the original source if no such pattern found.
    """
    pattern = r'''(?:^|\n)\s*\w+\s*=\s*(?:"{3}|'{3})(.*?)(?:"{3}|'{3})'''
    matches = re.findall(pattern, src, re.DOTALL)
    if matches:
        extracted = max(matches, key=len).strip()
        if len(extracted) > 20:
            return extracted
    return src


def extract_and_validate_ipynb(filepath: str) -> dict:
    """
    Opens a .ipynb file, extracts all code cells into a single
    string, strips Jupyter magic commands, unwraps any
    triple-quoted string wrappers, then validates syntax.

    Returns:
        {
            "status":  "success" | "error",
            "message": str,
            "code":    str | None
        }
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        return {
            "status":  "error",
            "message": "Invalid file format. Please upload a valid .ipynb file.",
            "code":    None,
        }
    except Exception as e:
        return {
            "status":  "error",
            "message": f"Could not read file: {e}",
            "code":    None,
        }

    clean_lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            # Strip Jupyter magic commands (!, %)
            if not line.strip().startswith(("!", "%")):
                clean_lines.append(line)
        clean_lines.append("\n\n")

    raw_code = "".join(clean_lines).strip()

    if not raw_code:
        return {
            "status":  "error",
            "message": "The uploaded notebook contains no code cells.",
            "code":    None,
        }

    # Unwrap triple-quoted string wrappers
    code = _unwrap_string_code(raw_code)

    # Syntax validation
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "status":  "error",
            "message": f"Syntax error on line {e.lineno}: {e.msg}. Fix your code and resubmit.",
            "code":    None,
        }

    return {
        "status":  "success",
        "message": "Valid code",
        "code":    code,
    }
