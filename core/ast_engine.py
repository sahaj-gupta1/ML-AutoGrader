"""
Dynamic AST Engine
==================
Replaces the old hardcoded ast_engine.py.

Key changes:
  - NO hardcoded rubric steps or subtasks
  - Parses .ipynb notebooks CELL BY CELL
  - Skips cells with SyntaxError / IndentationError gracefully
  - Returns a FactSheet (imports + function calls) that feeds
    into the LLM pipeline (ollama_client.py) for dynamic grading
  - Works with ANY rubric a teacher defines

Architecture (from samaeas20.ipynb):
  Layer 1 (THIS FILE) → FactSheet  (deterministic AST)
  Layer 2             → Annotated code  (LLM)
  Layer 3             → Execution timeline  (LLM)
  Layer 4             → Audited timeline  (LLM)
  Layer 5             → Scores + penalties  (Python math)
  Layer 6             → Feedback text  (LLM)
"""

import ast
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class CellResult:
    """Result of parsing a single notebook cell."""
    cell_index:   int
    source:       str          # raw cell source
    parseable:    bool         # False if SyntaxError / IndentationError
    error:        str = ""     # error message if not parseable
    imports:      list = field(default_factory=list)
    functions:    list = field(default_factory=list)


@dataclass
class FactSheet:
    """
    Aggregated facts extracted from all parseable cells.
    This is the output of Layer 1 — passed to the LLM layers.
    """
    imports:          list   # all unique imports across all cells
    functions:        list   # all unique function/method calls
    parseable_cells:  int    # how many cells parsed successfully
    failed_cells:     int    # how many cells had syntax errors
    failed_details:   list   # list of {cell_index, error} dicts
    combined_code:    str    # all parseable cell code joined (for LLM)
    cell_results:     list   # per-cell CellResult objects


# ─────────────────────────────────────────────────────────────
#  NOTEBOOK READER
# ─────────────────────────────────────────────────────────────

def _read_notebook_cells(ipynb_path: str) -> list[str]:
    """
    Read a .ipynb file and return a list of code cell sources.
    Each element is the raw source string of one code cell.
    Markdown / raw cells are skipped.
    """
    path = Path(ipynb_path)
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {ipynb_path}")

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        # source can be a list of lines or a single string
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            cells.append(source)

    return cells


# ─────────────────────────────────────────────────────────────
#  SINGLE-CELL AST PARSER
# ─────────────────────────────────────────────────────────────

def _parse_cell(cell_index: int, source: str) -> CellResult:
    """
    Parse a single cell. Returns a CellResult.

    Handles:
      - SyntaxError   (e.g. typos, incomplete code)
      - IndentationError (subclass of SyntaxError)
      - Any other parse-time exception

    Strategy: if the cell fails as-is, try dedenting it once
    (some students paste indented code into top-level cells).
    """
    result = CellResult(cell_index=cell_index, source=source, parseable=False)

    attempts = [source, textwrap.dedent(source)]

    for attempt in attempts:
        try:
            tree = ast.parse(attempt)
            result.parseable = True
            result.imports, result.functions = _extract_facts(tree)
            return result
        except (SyntaxError, IndentationError) as e:
            result.error = f"{type(e).__name__}: {e.msg} (line {e.lineno})"
        except Exception as e:
            result.error = f"ParseError: {str(e)}"

    return result  # parseable=False


def _extract_facts(tree: ast.AST) -> tuple[list, list]:
    """
    Walk an AST and extract:
      - imports (module names)
      - function/method calls (function or attribute names)
    """
    imports   = set()
    functions = set()

    for node in ast.walk(tree):
        # import pandas as pd  →  "pandas"
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

        # from sklearn.model_selection import train_test_split  →  "sklearn.model_selection"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

        # function call:  fit_transform(...)  →  "fit_transform"
        elif isinstance(node, ast.Call):
            if hasattr(node.func, "id"):          # bare function: read_csv()
                functions.add(node.func.id)
            elif hasattr(node.func, "attr"):       # method: scaler.fit_transform()
                functions.add(node.func.attr)

    return sorted(imports), sorted(functions)


# ─────────────────────────────────────────────────────────────
#  PUBLIC API — LAYER 1
# ─────────────────────────────────────────────────────────────

def extract_fact_sheet(ipynb_path: str) -> FactSheet:
    """
    Main entry point.

    Reads the .ipynb, parses each code cell independently,
    aggregates facts, and returns a FactSheet.

    Usage:
        from core.ast_engine import extract_fact_sheet
        facts = extract_fact_sheet("uploads/submissions/user_3_assign_1.ipynb")
    """
    cells = _read_notebook_cells(ipynb_path)

    all_imports   = set()
    all_functions = set()
    cell_results  = []
    failed_details = []

    parseable_code_parts = []

    for i, source in enumerate(cells):
        cr = _parse_cell(i, source)
        cell_results.append(cr)

        if cr.parseable:
            all_imports.update(cr.imports)
            all_functions.update(cr.functions)
            parseable_code_parts.append(f"# ── Cell {i} ──\n{source}")
        else:
            failed_details.append({
                "cell_index": i,
                "error": cr.error,
                "snippet": source[:120].replace("\n", " ") + ("..." if len(source) > 120 else "")
            })

    parseable  = sum(1 for cr in cell_results if cr.parseable)
    failed     = len(cell_results) - parseable

    return FactSheet(
        imports         = sorted(all_imports),
        functions       = sorted(all_functions),
        parseable_cells = parseable,
        failed_cells    = failed,
        failed_details  = failed_details,
        combined_code   = "\n\n".join(parseable_code_parts),
        cell_results    = cell_results,
    )


def fact_sheet_to_dict(fs: FactSheet) -> dict:
    """Convert FactSheet to a plain dict (for passing to LLM prompts)."""
    return {
        "imports":          fs.imports,
        "functions":        fs.functions,
        "parseable_cells":  fs.parseable_cells,
        "failed_cells":     fs.failed_cells,
        "failed_details":   fs.failed_details,
    }


# ─────────────────────────────────────────────────────────────
#  SCORING ENGINE  (Layer 5 — pure Python, no LLM)
# ─────────────────────────────────────────────────────────────

def calculate_final_scores(teacher_rubric: dict, student_timeline: list) -> dict:
    """
    Given a teacher rubric and the LLM-extracted execution timeline,
    compute scores, detect sequence violations, and identify missed steps.

    teacher_rubric format (same as notebook):
        {
            "data_loading":   {"points": 1, "depends_on": []},
            "basic_cleaning": {"points": 1, "depends_on": ["data_loading"]},
            ...
        }

    Returns:
        {
            "penalties":    [...],
            "missed_steps": [...],
            "scores":       {"step": score, ...},
            "total":        float,
        }
    """
    penalties    = []
    missed_steps = []
    final_scores = {key: 0.0 for key in teacher_rubric}

    # 1. Award base points for every detected step
    for step in teacher_rubric:
        if step in student_timeline:
            final_scores[step] = float(teacher_rubric[step]["points"])
        else:
            missed_steps.append(step)

    # 2. Check dependency order
    for current_idx, step in enumerate(student_timeline):
        rules = teacher_rubric.get(step, {})
        for required in rules.get("depends_on", []):
            if required in student_timeline:
                required_idx = student_timeline.index(required)
                if required_idx > current_idx:          # required came AFTER
                    penalty = rules["points"] / 2.0
                    final_scores[step] = max(0.0, final_scores[step] - penalty)
                    penalties.append(
                        f"VIOLATION: '{step}' executed BEFORE '{required}'. "
                        f"Penalty: -{penalty}"
                    )
            else:                                       # required is missing
                penalty = rules["points"] / 2.0
                final_scores[step] = max(0.0, final_scores[step] - penalty)
                penalties.append(
                    f"MISSING DEPENDENCY: '{step}' requires '{required}'. "
                    f"Penalty: -{penalty}"
                )

    # 3. Floor at 0
    for k in final_scores:
        if final_scores[k] < 0:
            final_scores[k] = 0.0

    return {
        "penalties":    penalties,
        "missed_steps": missed_steps,
        "scores":       final_scores,
        "total":        sum(final_scores.values()),
    }
