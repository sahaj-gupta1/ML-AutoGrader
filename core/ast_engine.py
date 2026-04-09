"""
Dynamic AST Engine
==================
Completely rubric-agnostic. Works with ANY step name the teacher defines.

Two-layer detection per step:
  Layer 1 — Known steps: use the curated signature map for rich sub-task detection
  Layer 2 — Unknown steps: derive keywords from step name, do semantic matching

No hardcoded step names are required for the system to work.
"""

import ast
import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    name:        str
    description: str
    required:    bool = True


@dataclass
class StepResult:
    step_name:          str
    detected:           bool
    completed_subtasks: list
    missed_subtasks:    list
    order_violations:   list
    first_line:         int
    partial_ratio:      float
    detection_method:   str   # 'known' | 'dynamic' | 'absent'


# ─────────────────────────────────────────────────────────────
#  KNOWN STEP SIGNATURES
#  Only used when teacher's step name matches exactly.
#  If teacher uses a different name → dynamic detection kicks in.
# ─────────────────────────────────────────────────────────────

KNOWN_STEP_SUBTASKS = {
    "data_loading": [
        SubTask("file_read",
                "Data loaded from a file (read_csv, read_excel, etc.)"),
    ],
    "eda_and_visualization": [
        SubTask("visual_or_stats",
                "At least one plot or statistical summary (plot, hist, describe, etc.)"),
    ],
    "basic_cleaning": [
        SubTask("missing_value_handling",
                "Missing values handled (dropna, fillna, or SimpleImputer)"),
        SubTask("duplicate_removal",
                "Duplicates removed (drop_duplicates)"),
        SubTask("target_transformation",
                "Target variable transformed or encoded (astype, map, LabelEncoder)"),
    ],
    "split_data": [
        SubTask("train_test_split_call", "Data split using train_test_split"),
        SubTask("xy_separation",         "Features (X) and target (y) separated"),
    ],
    "scaling_and_imputation": [
        SubTask("scaler_present",    "A scaler is used (StandardScaler, MinMaxScaler, etc.)"),
        SubTask("fit_on_train_only", "Scaler fitted on training data only (fit_transform on X_train)"),
    ],
    "model_training": [
        SubTask("model_instantiation", "A model class is instantiated"),
        SubTask("fit_call",            "Model is fitted with .fit()"),
    ],
    "hyperparameter_tuning": [
        SubTask("param_grid_defined", "Parameter grid defined (param_grid or param_distributions)"),
        SubTask("search_cv_used",     "GridSearchCV or RandomizedSearchCV used"),
    ],
    "evaluation": [
        SubTask("predict_call",  "Predictions generated with .predict()"),
        SubTask("metric_used",   "At least one evaluation metric used"),
    ],
}

KNOWN_SUBTASK_SIGNALS = {
    "file_read":              {"read_csv","read_excel","read_json","read_parquet","read_table","read_sql","load_dataset"},
    "visual_or_stats":        {"plot","show","hist","boxplot","heatmap","pairplot","countplot","describe","value_counts","corr","violinplot","figure","subplots","head","tail","info"},
    "missing_value_handling": {"dropna","fillna","SimpleImputer","KNNImputer","IterativeImputer"},
    "duplicate_removal":      {"drop_duplicates"},
    "target_transformation":  {"astype","map","LabelEncoder","OrdinalEncoder","replace"},
    "train_test_split_call":  {"train_test_split"},
    "xy_separation":          {"drop"},
    "scaler_present":         {"StandardScaler","MinMaxScaler","RobustScaler","Normalizer","MaxAbsScaler","PowerTransformer","QuantileTransformer"},
    "fit_on_train_only":      set(),  # argument analysis
    "model_instantiation":    {"LogisticRegression","RandomForestClassifier","SVC","DecisionTreeClassifier",
                               "KNeighborsClassifier","GradientBoostingClassifier","XGBClassifier",
                               "LinearRegression","Ridge","Lasso","ElasticNet","SVR",
                               "RandomForestRegressor","GradientBoostingRegressor","MLPClassifier",
                               "MLPRegressor","SGDClassifier","ExtraTreesClassifier","AdaBoostClassifier",
                               "GaussianNB","KMeans","DBSCAN","Sequential","Dense","Conv2D","LSTM"},
    "fit_call":               {"fit"},
    "param_grid_defined":     set(),  # variable name check
    "search_cv_used":         {"GridSearchCV","RandomizedSearchCV","BayesSearchCV","HalvingGridSearchCV"},
    "predict_call":           {"predict","predict_proba","evaluate"},
    "metric_used":            {"accuracy_score","f1_score","precision_score","recall_score",
                               "roc_auc_score","confusion_matrix","classification_report",
                               "mean_squared_error","mean_absolute_error","r2_score",
                               "log_loss","silhouette_score","score"},
}

# ─────────────────────────────────────────────────────────────
#  DYNAMIC KEYWORD MAP
#  Derives keywords from ANY step name a teacher might use.
#  Keywords → Python identifiers to look for in student code.
# ─────────────────────────────────────────────────────────────

DYNAMIC_KEYWORD_MAP = {
    # Data related
    "data":         ["read_csv","read_excel","read_json","load","open","dataset"],
    "load":         ["read_csv","read_excel","read_json","read_parquet","load_dataset"],
    "loading":      ["read_csv","read_excel","load","open"],
    "import":       ["read_csv","read_excel","read_json"],
    "ingest":       ["read_csv","read_excel","read_json","load"],

    # EDA / Exploration
    "eda":          ["describe","info","head","tail","plot","hist","corr","value_counts","heatmap","boxplot"],
    "exploration":  ["describe","info","head","tail","plot","hist","corr"],
    "exploratory":  ["describe","info","head","tail","plot","hist","corr","value_counts"],
    "analysis":     ["describe","info","corr","value_counts","groupby","plot"],
    "visual":       ["plot","show","hist","boxplot","scatter","heatmap","seaborn","matplotlib","figure"],
    "visualization":["plot","show","hist","boxplot","scatter","heatmap","pairplot","figure"],
    "statistics":   ["describe","mean","std","corr","value_counts","groupby"],

    # Cleaning
    "clean":        ["dropna","fillna","drop_duplicates","replace","strip","astype"],
    "cleaning":     ["dropna","fillna","drop_duplicates","replace","strip","astype"],
    "preprocess":   ["dropna","fillna","drop_duplicates","LabelEncoder","get_dummies","astype"],
    "preprocessing":["dropna","fillna","drop_duplicates","LabelEncoder","get_dummies","StandardScaler","astype"],
    "wrangling":    ["dropna","fillna","drop_duplicates","replace","astype","rename"],
    "missing":      ["dropna","fillna","SimpleImputer","KNNImputer","isnull","isna"],
    "imputation":   ["fillna","SimpleImputer","KNNImputer","IterativeImputer"],
    "impute":       ["fillna","SimpleImputer","KNNImputer"],
    "duplicate":    ["drop_duplicates","duplicated"],
    "outlier":      ["quantile","IQR","clip","zscore","boxplot","isoutlier"],
    "encoding":     ["LabelEncoder","OrdinalEncoder","OneHotEncoder","get_dummies","astype"],
    "encode":       ["LabelEncoder","OrdinalEncoder","OneHotEncoder","get_dummies"],

    # Feature engineering
    "feature":      ["drop","get_dummies","LabelEncoder","StandardScaler","corr","SelectKBest"],
    "engineering":  ["drop","get_dummies","LabelEncoder","StandardScaler","corr","SelectKBest","fillna"],
    "selection":    ["SelectKBest","RFE","corr","drop","feature_importances_"],
    "extraction":   ["PCA","TruncatedSVD","SelectKBest"],

    # Scaling
    "scaling":      ["StandardScaler","MinMaxScaler","RobustScaler","Normalizer","fit_transform","transform"],
    "scale":        ["StandardScaler","MinMaxScaler","RobustScaler","fit_transform","transform"],
    "normaliz":     ["Normalizer","MinMaxScaler","normalize"],
    "standardiz":   ["StandardScaler","fit_transform"],

    # Splitting
    "split":        ["train_test_split"],
    "splitting":    ["train_test_split"],
    "partition":    ["train_test_split"],
    "train":        ["train_test_split","fit"],
    "test":         ["train_test_split","predict","score"],

    # Model
    "model":        ["fit","LogisticRegression","RandomForest","SVC","Sequential","LinearRegression"],
    "training":     ["fit"],
    "fitting":      ["fit"],
    "algorithm":    ["fit","LogisticRegression","RandomForest","SVC","LinearRegression","Ridge"],
    "architecture": ["Sequential","Dense","Conv2D","LSTM","add","layers"],
    "neural":       ["Sequential","Dense","Conv2D","LSTM","keras","tensorflow","torch"],
    "deep":         ["Sequential","Dense","Conv2D","LSTM","keras","tensorflow"],
    "network":      ["Sequential","Dense","Conv2D","LSTM","add"],
    "compilation":  ["compile","optimizer","loss"],
    "compile":      ["compile","optimizer"],
    "regression":   ["LinearRegression","Ridge","Lasso","SVR","GradientBoostingRegressor","fit"],
    "classification":["LogisticRegression","SVC","RandomForestClassifier","fit"],
    "clustering":   ["KMeans","DBSCAN","AgglomerativeClustering","fit"],

    # Tuning
    "hyperparameter":["GridSearchCV","RandomizedSearchCV","param_grid","best_params_"],
    "tuning":       ["GridSearchCV","RandomizedSearchCV","param_grid","best_params_"],
    "optimization": ["GridSearchCV","RandomizedSearchCV","param_grid","optimize"],
    "search":       ["GridSearchCV","RandomizedSearchCV","param_grid"],

    # Evaluation
    "evaluation":   ["predict","accuracy_score","f1_score","r2_score","confusion_matrix","classification_report"],
    "evaluate":     ["predict","accuracy_score","f1_score","r2_score","evaluate","score"],
    "validation":   ["cross_val_score","KFold","StratifiedKFold","predict","accuracy_score"],
    "metric":       ["accuracy_score","f1_score","r2_score","mean_squared_error","confusion_matrix"],
    "performance":  ["accuracy_score","f1_score","r2_score","mean_squared_error","score"],
    "testing":      ["predict","accuracy_score","score"],
    "predict":      ["predict","predict_proba"],
    "accuracy":     ["accuracy_score","score"],
}


def _get_dynamic_keywords(step_name: str) -> set:
    """
    Derive Python identifiers to look for from any step name.
    Splits the step name into words and looks them up in DYNAMIC_KEYWORD_MAP.
    Works for steps like 'Data Cleaning', 'data_cleaning', 'Scaling & Encoding' etc.
    """
    # Normalise: lowercase, split on spaces, underscores, &, /, -
    words = re.split(r'[\s_&/\-]+', step_name.lower())
    keywords = set()

    for word in words:
        word = word.strip()
        if not word:
            continue
        # Exact match
        if word in DYNAMIC_KEYWORD_MAP:
            keywords.update(DYNAMIC_KEYWORD_MAP[word])
        else:
            # Partial match — check if word starts with any key
            for key, vals in DYNAMIC_KEYWORD_MAP.items():
                if word.startswith(key) or key.startswith(word):
                    keywords.update(vals)

    return keywords


# ─────────────────────────────────────────────────────────────
#  AST WALKERS
# ─────────────────────────────────────────────────────────────

def analyse_code(code: str) -> dict:
    """Extract all identifiers from student code. No execution."""
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
    """{ call_name -> first line it is called }. Import lines excluded."""
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
        if name not in index:
            index[name] = node.lineno
    return index


def _build_assign_index(code: str) -> dict:
    """{ variable_name -> first line of assignment }."""
    index = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return index
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in index:
                    index[target.id] = node.lineno
    return index


def _check_fit_transform_on_train(code: str) -> bool:
    """Checks if fit_transform was called on training data (not full X)."""
    TRAIN_HINTS = {"train","_train","X_train","x_train","xtrain","X_tr"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "fit_transform"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        arg_name = first_arg.id if isinstance(first_arg, ast.Name) else ""
        if any(hint.lower() in arg_name.lower() for hint in TRAIN_HINTS):
            return True
        if arg_name in ("X", "x"):
            return False
    return False


# ─────────────────────────────────────────────────────────────
#  KNOWN STEP DETECTION (rich sub-task level)
# ─────────────────────────────────────────────────────────────

def _check_known_step(step_name: str, code: str, facts: dict,
                      call_index: dict, assign_index: dict) -> tuple:
    """
    Detailed sub-task checking for known step names.
    Returns (completed_subtasks, missed_subtasks, first_line)
    """
    subtasks = KNOWN_STEP_SUBTASKS.get(step_name, [])
    completed, missed, all_lines = [], [], []

    for subtask in subtasks:
        signals  = KNOWN_SUBTASK_SIGNALS.get(subtask.name, set())
        detected = False

        if subtask.name == "fit_on_train_only":
            scaler_exists = any(
                s in facts["functions"] or s in facts["classes"]
                for s in KNOWN_SUBTASK_SIGNALS["scaler_present"]
            )
            detected = _check_fit_transform_on_train(code) if scaler_exists else False

        elif subtask.name == "param_grid_defined":
            detected = ("param_grid" in assign_index or
                        "param_distributions" in assign_index or
                        "param_grid" in facts["variables"])

        elif subtask.name == "xy_separation":
            detected = ("X" in assign_index or "x" in assign_index or
                        "features" in assign_index or
                        "drop" in facts["functions"])

        elif subtask.name == "target_transformation":
            detected = any(s in facts["functions"] or s in facts["classes"]
                           for s in signals)
            if not detected:
                detected = "astype" in facts["functions"]
        else:
            detected = any(s in facts["functions"] or s in facts["classes"]
                           for s in signals)

        if detected:
            completed.append(subtask)
            for sig in signals:
                if sig in call_index:
                    all_lines.append(call_index[sig])
        else:
            missed.append(subtask)

    first_line = min(all_lines) if all_lines else None
    return completed, missed, first_line


# ─────────────────────────────────────────────────────────────
#  DYNAMIC STEP DETECTION (for any unknown step name)
# ─────────────────────────────────────────────────────────────

def _check_dynamic_step(step_name: str, facts: dict,
                        call_index: dict) -> tuple:
    """
    For any step name the teacher defines, derive keywords and
    check if any matching identifiers exist in student code.
    Returns (detected: bool, evidence: list, first_line: int|None)
    """
    keywords = _get_dynamic_keywords(step_name)

    if not keywords:
        return False, [], None

    evidence   = []
    all_lines  = []

    all_identifiers = facts["functions"] | facts["classes"] | facts["imports"]

    for kw in keywords:
        # Check exact match in any identifier set
        if kw in all_identifiers:
            evidence.append(kw)
            if kw in call_index:
                all_lines.append(call_index[kw])
        else:
            # Check if any import contains this keyword (e.g. "sklearn" in "sklearn.preprocessing")
            for imp in facts["imports"]:
                if kw.lower() in imp.lower():
                    evidence.append(f"import:{imp}")
                    break

    first_line = min(all_lines) if all_lines else None
    return len(evidence) > 0, evidence, first_line


# ─────────────────────────────────────────────────────────────
#  MAIN: DETECT AND EVALUATE ALL STEPS
# ─────────────────────────────────────────────────────────────

CROSS_STEP_ORDER_RULES = [
    ("data_loading",    "basic_cleaning",
     "'basic_cleaning' must come after 'data_loading'"),
    ("basic_cleaning",  "split_data",
     "'basic_cleaning' should come before 'split_data'"),
    ("split_data",      "scaling_and_imputation",
     "'scaling_and_imputation' must come AFTER 'split_data' to avoid data leakage"),
    ("split_data",      "model_training",
     "'model_training' must come after 'split_data'"),
    ("model_training",  "evaluation",
     "'evaluation' must come after 'model_training'"),
]

SEARCH_CLASSES = {
    "GridSearchCV","RandomizedSearchCV","BayesSearchCV","HalvingGridSearchCV"
}


def detect_and_evaluate_steps(
    code: str,
    rubric: dict,
    teacher_custom_subtasks: dict = None,
) -> tuple:
    """
    Main entry point. Works with ANY rubric step names.

    For each step:
    - If step name is in KNOWN_STEP_SUBTASKS → rich sub-task detection
    - Otherwise → dynamic keyword detection from the step name

    Returns (results dict, cross_violations list)
    """
    if teacher_custom_subtasks is None:
        teacher_custom_subtasks = {}

    facts        = analyse_code(code)
    call_index   = _build_call_index(code)
    assign_index = _build_assign_index(code)

    results           = {}
    step_first_lines  = {}

    for step_name in rubric:
        # Normalise step name for lookup (lowercase, underscores)
        normalised = step_name.lower().replace(" ", "_").replace("&","and").replace("/","_")

        # ── Layer 1: known step ──────────────────────────────
        if normalised in KNOWN_STEP_SUBTASKS:
            completed, missed, first_line = _check_known_step(
                normalised, code, facts, call_index, assign_index
            )

            # EDA fallback: matplotlib/seaborn import counts
            if normalised == "eda_and_visualization" and not completed:
                eda_imports = {"matplotlib","seaborn","plotly"}
                for imp in eda_imports:
                    if any(f.startswith(imp) for f in facts["imports"]):
                        vis_task = SubTask("visual_or_stats",
                                           "Visualization library imported")
                        completed.append(vis_task)
                        missed = [m for m in missed if m.name != "visual_or_stats"]
                        break

            # scaling special case
            if normalised == "scaling_and_imputation":
                scaler_present = any(
                    t.name == "scaler_present" for t in completed
                )
                if not scaler_present:
                    # No scaler at all — treat as absent
                    result = StepResult(
                        step_name=step_name, detected=False,
                        completed_subtasks=[], missed_subtasks=[s.description for s in missed],
                        order_violations=[], first_line=99999,
                        partial_ratio=0.0, detection_method="known"
                    )
                    results[step_name] = result
                    continue

            total = len(completed) + len(missed)
            ratio = len(completed) / total if total > 0 else (1.0 if completed else 0.0)
            detected = len(completed) > 0

            result = StepResult(
                step_name          = step_name,
                detected           = detected,
                completed_subtasks = [s.name for s in completed],
                missed_subtasks    = [s.description for s in missed],
                order_violations   = [],
                first_line         = first_line or 99999,
                partial_ratio      = ratio,
                detection_method   = "known",
            )

        # ── Layer 2: dynamic detection ───────────────────────
        else:
            detected, evidence, first_line = _check_dynamic_step(
                step_name, facts, call_index
            )

            result = StepResult(
                step_name          = step_name,
                detected           = detected,
                completed_subtasks = evidence if detected else [],
                missed_subtasks    = [] if detected else
                                     [f"No evidence of '{step_name}' found in code"],
                order_violations   = [],
                first_line         = first_line or 99999,
                partial_ratio      = 1.0 if detected else 0.0,
                detection_method   = "dynamic",
            )

        results[step_name] = result
        if result.first_line < 99999:
            step_first_lines[step_name] = result.first_line

    # Fix split_data ordering — use train_test_split line only
    for step_name in results:
        norm = step_name.lower().replace(" ","_")
        if "split" in norm and "train_test_split" in call_index:
            results[step_name].first_line = call_index["train_test_split"]
            step_first_lines[step_name]   = call_index["train_test_split"]

    # Cross-step ordering violations (only for known step names)
    cross_violations = []
    for step_a_key, step_b_key, msg in CROSS_STEP_ORDER_RULES:
        # Find matching step names in rubric (case-insensitive)
        step_a_match = next(
            (s for s in rubric
             if s.lower().replace(" ","_").replace("&","and") == step_a_key), None
        )
        step_b_match = next(
            (s for s in rubric
             if s.lower().replace(" ","_").replace("&","and") == step_b_key), None
        )
        if step_a_match and step_b_match:
            if step_a_match in step_first_lines and step_b_match in step_first_lines:
                if step_first_lines[step_a_match] > step_first_lines[step_b_match]:
                    cross_violations.append(msg)

    # GridSearchCV fix — align hyperparameter_tuning and model_training
    if any(s in facts["functions"] or s in facts["classes"] for s in SEARCH_CLASSES):
        tuning_step  = next((s for s in results
                             if "tuning" in s.lower() or "hyperparameter" in s.lower()), None)
        training_step = next((s for s in results
                              if "training" in s.lower() or s.lower().replace(" ","_") == "model_training"), None)
        if tuning_step and training_step and \
                tuning_step in step_first_lines and training_step in step_first_lines:
            shared = min(step_first_lines[tuning_step], step_first_lines[training_step])
            step_first_lines[tuning_step]  = shared
            step_first_lines[training_step] = shared

    return results, cross_violations


def detect_steps(code: str, facts: dict, rubric_keys: list) -> list:
    """Legacy compatibility — returns ordered list of detected step names."""
    mini_rubric = {k: {"points": 1, "depends_on": []} for k in rubric_keys}
    results, _ = detect_and_evaluate_steps(code, mini_rubric)
    step_lines  = {s: r.first_line for s, r in results.items()
                   if r.detected and r.first_line < 99999}
    return sorted(step_lines, key=lambda s: step_lines[s])
