"""
Enhanced AST Engine
===================
Goes beyond binary step detection to check:
  1. Which critical sub-tasks were completed within each step
  2. Whether sub-tasks were done in the correct order
  3. Argument-level checks (e.g. fit_transform on X_train vs full X)

Scoring is proportional:
  step_score = (completed_subtasks / total_subtasks) * step_points
"""

import ast
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A single critical requirement within a rubric step."""
    name:        str            # e.g. "missing_value_handling"
    description: str            # shown in feedback
    required:    bool = True    # if False, optional bonus check


@dataclass
class StepResult:
    """Result of checking one rubric step."""
    step_name:          str
    detected:           bool
    completed_subtasks: list   # list of SubTask names that passed
    missed_subtasks:    list   # list of SubTask names that failed
    order_violations:   list   # list of violation message strings
    first_line:         int    # line number of first relevant call
    partial_ratio:      float  # completed / total (0.0 to 1.0)


# ─────────────────────────────────────────────────────────────
#  CRITICAL SUB-TASK DEFINITIONS
#
#  Only the sub-tasks that actually matter for ML correctness.
#  info(), describe(), head() etc. are excluded — they don't
#  affect the model or data pipeline.
# ─────────────────────────────────────────────────────────────

STEP_SUBTASKS = {
    "data_loading": [
        SubTask("file_read",
                "Data loaded from a file (read_csv, read_excel, etc.)"),
    ],

    "eda_and_visualization": [
        SubTask("visual_or_stats",
                "At least one plot or statistical summary (plot, hist, "
                "describe, value_counts, corr, boxplot, etc.)"),
    ],

    "basic_cleaning": [
        SubTask("missing_value_handling",
                "Missing values handled (dropna, fillna, or SimpleImputer)"),
        SubTask("duplicate_removal",
                "Duplicates removed (drop_duplicates)"),
        SubTask("target_transformation",
                "Target variable transformed or encoded (astype, map, "
                "LabelEncoder, or binary threshold applied)"),
    ],

    "split_data": [
        SubTask("train_test_split_call",
                "Data split into train/test sets (train_test_split)"),
        SubTask("xy_separation",
                "Features (X) and target (y) separated before splitting"),
    ],

    "scaling_and_imputation": [
        SubTask("scaler_present",
                "A scaler is used (StandardScaler, MinMaxScaler, etc.)"),
        SubTask("fit_on_train_only",
                "Scaler fitted on training data only, not full dataset "
                "(fit_transform on X_train, not on X)"),
    ],

    "model_training": [
        SubTask("model_instantiation",
                "A model class is instantiated (LogisticRegression(), "
                "RandomForestClassifier(), etc.)"),
        SubTask("fit_call",
                "Model is fitted with .fit() on training data"),
    ],

    "hyperparameter_tuning": [
        SubTask("param_grid_defined",
                "Parameter grid defined (param_grid or param_distributions)"),
        SubTask("search_cv_used",
                "GridSearchCV or RandomizedSearchCV used to search"),
    ],

    "evaluation": [
        SubTask("predict_call",
                "Predictions generated with .predict()"),
        SubTask("metric_used",
                "At least one evaluation metric used (accuracy_score, "
                "f1_score, classification_report, r2_score, etc.)"),
    ],
}

# ─────────────────────────────────────────────────────────────
#  CORRECT ORDERING WITHIN STEPS
#
#  Format: { step: [(task_A, task_B, "violation message if A after B") ] }
#  Meaning: task_A must come BEFORE task_B
# ─────────────────────────────────────────────────────────────

SUBTASK_ORDER_RULES = {
    # basic_cleaning order (missing before duplicates) removed from penalties
    # — kept as scientific note only, not penalised
    "scaling_and_imputation": [
        (
            "fit_on_train_only",
            "scaler_present",
            None   # not a violation, handled in scoring_engine
        ),
    ],
}

# Cross-step ordering rules — (step_A must come before step_B)
CROSS_STEP_ORDER_RULES = [
    ("basic_cleaning",        "split_data",
     "'basic_cleaning' should come before 'split_data'"),
    ("split_data",            "scaling_and_imputation",
     "'scaling_and_imputation' must come AFTER 'split_data' to avoid "
     "data leakage — scaler must be fitted on training data only"),
    ("split_data",            "model_training",
     "'model_training' must come after 'split_data'"),
    ("model_training",        "evaluation",
     "'evaluation' must come after 'model_training'"),
    ("data_loading",          "basic_cleaning",
     "'basic_cleaning' must come after 'data_loading'"),
]

# ─────────────────────────────────────────────────────────────
#  DETECTION HELPERS — what identifiers signal each sub-task
# ─────────────────────────────────────────────────────────────

# Maps sub-task name → set of function/method call names that prove it
SUBTASK_SIGNALS = {
    # data_loading
    "file_read": {
        "read_csv", "read_excel", "read_json", "read_parquet",
        "read_table", "read_sql", "load_dataset", "read_feather",
    },

    # eda
    "visual_or_stats": {
        "plot", "show", "hist", "boxplot", "heatmap", "pairplot",
        "countplot", "barplot", "scatterplot", "describe",
        "value_counts", "corr", "violinplot", "scatter", "bar",
        "figure", "subplots",
    },

    # basic_cleaning
    "missing_value_handling": {
        "dropna", "fillna", "SimpleImputer", "KNNImputer",
        "IterativeImputer",
    },
    "duplicate_removal": {
        "drop_duplicates",
    },
    "target_transformation": {
        "astype", "map", "LabelEncoder", "OrdinalEncoder",
        "replace",
    },

    # split_data
    "train_test_split_call": {
        "train_test_split",
    },
    "xy_separation": {
        "drop",   # X = df.drop('target', axis=1)
    },

    # scaling
    "scaler_present": {
        "StandardScaler", "MinMaxScaler", "RobustScaler",
        "Normalizer", "MaxAbsScaler", "PowerTransformer",
        "QuantileTransformer",
    },
    # fit_on_train_only is checked via argument analysis (see below)
    "fit_on_train_only": set(),  # handled separately

    # model_training
    "model_instantiation": {
        "LogisticRegression", "RandomForestClassifier",
        "SVC", "DecisionTreeClassifier", "KNeighborsClassifier",
        "GradientBoostingClassifier", "XGBClassifier",
        "LGBMClassifier", "LinearRegression", "Ridge", "Lasso",
        "ElasticNet", "SVR", "RandomForestRegressor",
        "GradientBoostingRegressor", "MLPClassifier",
        "MLPRegressor", "SGDClassifier", "ExtraTreesClassifier",
        "AdaBoostClassifier", "GaussianNB", "KMeans", "DBSCAN",
    },
    "fit_call": {
        "fit",
    },

    # hyperparameter_tuning
    "param_grid_defined": set(),  # detected via variable name (param_grid)
    "search_cv_used": {
        "GridSearchCV", "RandomizedSearchCV", "BayesSearchCV",
        "HalvingGridSearchCV",
    },

    # evaluation
    "predict_call": {
        "predict", "predict_proba",
    },
    "metric_used": {
        "accuracy_score", "f1_score", "precision_score",
        "recall_score", "roc_auc_score", "confusion_matrix",
        "classification_report", "mean_squared_error",
        "mean_absolute_error", "r2_score", "log_loss",
        "balanced_accuracy_score", "silhouette_score",
        "score",
    },
}

# Import signals — confirms a step is intended even without a direct call
STEP_IMPORT_SIGNALS = {
    "data_loading":           {"pandas", "datasets"},
    "eda_and_visualization":  {"matplotlib", "matplotlib.pyplot", "seaborn", "plotly"},
    "basic_cleaning":         {"sklearn.preprocessing", "sklearn.impute"},
    "split_data":             {"sklearn.model_selection"},
    "scaling_and_imputation": {"sklearn.preprocessing", "sklearn.impute"},
    "model_training":         {"sklearn.linear_model", "sklearn.ensemble",
                               "sklearn.svm", "sklearn.tree",
                               "sklearn.neighbors", "xgboost", "lightgbm"},
    "hyperparameter_tuning":  {"sklearn.model_selection"},
    "evaluation":             {"sklearn.metrics"},
}


# ─────────────────────────────────────────────────────────────
#  AST WALKERS
# ─────────────────────────────────────────────────────────────

def analyse_code(code: str) -> dict:
    """Extract imports, function calls, classes, variables from code."""
    facts = {
        "imports":   set(),
        "functions": set(),
        "classes":   set(),
        "variables": set(),
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return facts

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts["imports"].add(alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            for i in range(1, len(parts) + 1):
                facts["imports"].add(".".join(parts[:i]))
            for alias in node.names:
                facts["functions"].add(alias.name)
                facts["classes"].add(alias.name)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                facts["functions"].add(node.func.id)
                facts["classes"].add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                facts["functions"].add(node.func.attr)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    facts["variables"].add(target.id)

    return facts


def _build_call_index(code: str) -> dict:
    """
    Returns { call_name -> [line_numbers] } for every function/method call.
    Note: returns ALL occurrences (not just first) for ordering analysis.
    """
    index = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return index

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        index.setdefault(name, []).append(node.lineno)

    return index


def _build_assign_index(code: str) -> dict:
    """Returns { variable_name -> first_line } for assignments."""
    index = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return index

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id not in index:
                        index[target.id] = node.lineno

    return index


def _check_fit_transform_on_train(code: str) -> bool:
    """
    Returns True if fit_transform is called with a variable that looks
    like training data (X_train, train_X, etc.) rather than the full X.

    This is the data leakage check for scaling.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    TRAIN_HINTS = {"train", "_train", "X_train", "train_X",
                   "x_train", "xtrain", "X_tr", "x_tr"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Looking for .fit_transform(...)
        if not (isinstance(node.func, ast.Attribute) and
                node.func.attr == "fit_transform"):
            continue
        if not node.args:
            continue

        # Check the first argument name
        first_arg = node.args[0]
        arg_name = ""
        if isinstance(first_arg, ast.Name):
            arg_name = first_arg.id
        elif isinstance(first_arg, ast.Subscript):
            # e.g. df[features]
            arg_name = ""

        # Accept if argument contains a train indicator
        if any(hint.lower() in arg_name.lower() for hint in TRAIN_HINTS):
            return True

        # Also accept if it's clearly NOT the full X (single letter var)
        # Single letter 'X' without 'train' = full dataset = leakage
        if arg_name == "X" or arg_name == "x":
            return False

    # If fit_transform exists but we couldn't determine — assume leakage risk
    return False


# ─────────────────────────────────────────────────────────────
#  CORE: CHECK ALL SUB-TASKS FOR ONE STEP
# ─────────────────────────────────────────────────────────────

def _check_subtasks(
    step_name: str,
    code: str,
    facts: dict,
    call_index: dict,
    assign_index: dict,
    extra_required: list = None,    # teacher-added custom sub-tasks
) -> tuple:
    """
    Check all critical sub-tasks for a given step.
    Returns (completed, missed, first_line_of_step)
    """
    subtasks = STEP_SUBTASKS.get(step_name, [])

    # Add teacher-defined custom sub-tasks if provided
    if extra_required:
        for custom in extra_required:
            subtasks = subtasks + [SubTask(
                name        = f"custom_{custom.replace(' ', '_')}",
                description = custom,
            )]

    completed  = []
    missed     = []
    all_lines  = []

    for subtask in subtasks:
        signals  = SUBTASK_SIGNALS.get(subtask.name, set())
        detected = False

        # Special case: fit_on_train_only requires argument analysis
        if subtask.name == "fit_on_train_only":
            # Only check if a scaler exists at all
            scaler_exists = any(
                s in facts["functions"] or s in facts["classes"]
                for s in SUBTASK_SIGNALS["scaler_present"]
            )
            if scaler_exists:
                detected = _check_fit_transform_on_train(code)
            else:
                detected = False

        # Special case: param_grid is a variable name
        elif subtask.name == "param_grid_defined":
            detected = (
                "param_grid"          in assign_index or
                "param_distributions" in assign_index or
                "param_grid"          in facts["variables"] or
                "param_distributions" in facts["variables"]
            )

        # Special case: xy_separation — look for X/y assignment or drop
        # Note: we do NOT use drop call line for ordering because drop
        # also appears in cleaning steps — use assign_index for X/y
        elif subtask.name == "xy_separation":
            detected = (
                "X" in assign_index or "x" in assign_index or
                "features" in assign_index or
                "drop" in facts["functions"]
            )

        # Special case: target_transformation — astype(int) or similar
        elif subtask.name == "target_transformation":
            detected = any(
                s in facts["functions"] or s in facts["classes"]
                for s in signals
            )
            # Also check for boolean threshold pattern (quality >= 7).astype(int)
            if not detected:
                detected = "astype" in facts["functions"]

        # General case: check if any signal function/class was called
        else:
            detected = any(
                s in facts["functions"] or
                s in facts["classes"]
                for s in signals
            )

        if detected:
            completed.append(subtask)
            # Collect line numbers for ordering
            for sig in signals:
                if sig in call_index:
                    all_lines.extend(call_index[sig])
        else:
            missed.append(subtask)

    first_line = min(all_lines) if all_lines else None
    return completed, missed, first_line


# ─────────────────────────────────────────────────────────────
#  MAIN: DETECT AND EVALUATE ALL STEPS
# ─────────────────────────────────────────────────────────────

def detect_and_evaluate_steps(
    code: str,
    rubric: dict,
    teacher_custom_subtasks: dict = None,
) -> tuple:
    """
    Main entry point. Analyses code against rubric and returns:
      - results: { step_name -> StepResult }
      - cross_step_violations: list of violation message strings

    Args:
        code                  : extracted student Python code
        rubric                : { step_name: { points, depends_on } }
        teacher_custom_subtasks: { step_name: ["custom requirement 1", ...] }
    """
    if teacher_custom_subtasks is None:
        teacher_custom_subtasks = {}

    facts        = analyse_code(code)
    call_index   = _build_call_index(code)
    assign_index = _build_assign_index(code)

    # EDA fallback: matplotlib/seaborn import counts even without plot call
    eda_import_present = any(
        imp in facts["imports"] or
        any(imp2.startswith(imp) for imp2 in facts["imports"])
        for imp in {"matplotlib", "seaborn", "plotly"}
    )

    results = {}
    step_first_lines = {}   # { step_name -> first line } for cross-step ordering

    for step_name in rubric:
        extra = teacher_custom_subtasks.get(step_name, [])

        completed, missed, first_line = _check_subtasks(
            step_name    = step_name,
            code         = code,
            facts        = facts,
            call_index   = call_index,
            assign_index = assign_index,
            extra_required = extra,
        )

        # EDA special case: if matplotlib/seaborn imported, count visual as done
        if step_name == "eda_and_visualization" and not completed:
            if eda_import_present:
                vis_subtask = SubTask(
                    "visual_or_stats",
                    "Visualization library imported (matplotlib/seaborn)"
                )
                completed.append(vis_subtask)
                missed = [m for m in missed if m.name != "visual_or_stats"]

        total_subtasks = len(completed) + len(missed)

        # Special case: scaling_and_imputation is only "detected"
        # if the scaler itself is present. fit_on_train_only alone
        # cannot exist without a scaler — but be explicit here.
        if step_name == "scaling_and_imputation":
            detected = "scaler_present" in [s.name for s in completed]
        else:
            detected = len(completed) > 0

        if total_subtasks > 0:
            ratio = len(completed) / total_subtasks
        else:
            ratio = 1.0 if detected else 0.0

        # Within-step ordering violations
        order_violations = []
        rules = SUBTASK_ORDER_RULES.get(step_name, [])
        completed_names = {s.name for s in completed}
        for task_a, task_b, msg in rules:
            if msg and task_a in completed_names and task_b in completed_names:
                # Check actual line order
                lines_a = []
                lines_b = []
                for sig in SUBTASK_SIGNALS.get(task_a, set()):
                    lines_a.extend(call_index.get(sig, []))
                for sig in SUBTASK_SIGNALS.get(task_b, set()):
                    lines_b.extend(call_index.get(sig, []))
                if lines_a and lines_b:
                    if min(lines_a) > min(lines_b):
                        order_violations.append(msg)

        result = StepResult(
            step_name          = step_name,
            detected           = detected,
            completed_subtasks = [s.name for s in completed],
            missed_subtasks    = [s.description for s in missed],
            order_violations   = order_violations,
            first_line         = first_line or 99999,
            partial_ratio      = ratio,
        )
        results[step_name] = result

        if first_line:
            step_first_lines[step_name] = first_line

    # Fix split_data ordering BEFORE cross violation check
    # xy_separation uses "drop" which also appears in cleaning cells
    # so we override split_data first_line with the actual train_test_split call
    if "split_data" in results:
        tts_lines = call_index.get("train_test_split", [])
        if tts_lines:
            tts_line = min(tts_lines) if isinstance(tts_lines, list) else tts_lines
            results["split_data"].first_line = tts_line
            step_first_lines["split_data"]   = tts_line

    # Cross-step ordering violations
    cross_violations = []
    for step_a, step_b, msg in CROSS_STEP_ORDER_RULES:
        if step_a in step_first_lines and step_b in step_first_lines:
            if step_first_lines[step_a] > step_first_lines[step_b]:
                # step_a came AFTER step_b — violation
                if step_a in results and step_b in results:
                    cross_violations.append(msg)

    # GridSearchCV fix — tuning and training are the same call
    SEARCH_CLASSES = {
        "GridSearchCV", "RandomizedSearchCV",
        "BayesSearchCV", "HalvingGridSearchCV",
    }
    if any(s in facts["functions"] or s in facts["classes"]
           for s in SEARCH_CLASSES):
        if "hyperparameter_tuning" in step_first_lines and \
                "model_training" in step_first_lines:
            shared = min(
                step_first_lines["hyperparameter_tuning"],
                step_first_lines["model_training"],
            )
            step_first_lines["hyperparameter_tuning"] = shared
            step_first_lines["model_training"]         = shared

    return results, cross_violations


# ─────────────────────────────────────────────────────────────
#  LEGACY COMPATIBILITY
#  detect_steps() still works for any code that calls it directly
# ─────────────────────────────────────────────────────────────

def detect_steps(code: str, facts: dict, rubric_keys: list) -> list:
    """
    Returns chronologically ordered list of detected step names.
    Used by the scoring engine for dependency ordering.
    """
    call_index   = _build_call_index(code)
    assign_index = _build_assign_index(code)
    step_lines   = {}

    # Build a mini rubric for detect_and_evaluate
    mini_rubric = {k: {"points": 1, "depends_on": []} for k in rubric_keys}
    results, _ = detect_and_evaluate_steps(code, mini_rubric)

    for step_name, result in results.items():
        if result.detected and result.first_line < 99999:
            step_lines[step_name] = result.first_line

    return sorted(step_lines, key=lambda s: step_lines[s])
