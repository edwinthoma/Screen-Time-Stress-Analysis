import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    TEXT_COLUMN_CONFIG,
    ensure_project_dirs,
    get_dataset_path,
)


logger = logging.getLogger("ml_system")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def load_dataset() -> pd.DataFrame:
    ensure_project_dirs()
    dataset_path = get_dataset_path()
    if not dataset_path.exists():
        logger.warning("Dataset not found at %s", dataset_path)
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    logger.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def _detect_text_column(df: pd.DataFrame, target_col: str) -> Optional[str]:
    candidate = TEXT_COLUMN_CONFIG
    if candidate and "," not in candidate and candidate in df.columns:
        logger.info("Using configured text column: %s", candidate)
        return candidate

    object_cols = [c for c in df.columns if df[c].dtype == "object" and c != target_col]
    if not object_cols:
        logger.info("No textual columns detected.")
        return None

    best_col = None
    best_score = -1
    for col in object_cols:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        avg_len = sample.map(len).mean()
        unique_ratio = sample.nunique() / max(len(sample), 1)
        score = avg_len * unique_ratio
        if score > best_score:
            best_score = score
            best_col = col
    if best_col:
        logger.info("Auto-detected text column: %s", best_col)
    return best_col


def analyze_schema(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    logger.info("Analyzing dataframe schema")
    dtypes = df.dtypes.to_dict()
    logger.info("Column dtypes: %s", dtypes)

    id_like_cols: List[str] = []
    high_null_cols: List[str] = []
    constant_cols: List[str] = []

    for col in df.columns:
        series = df[col]
        if series.isnull().mean() > 0.5:
            high_null_cols.append(col)
        if series.nunique(dropna=True) <= 1:
            constant_cols.append(col)
        if series.nunique(dropna=True) >= 0.95 * len(series):
            id_like_cols.append(col)

    cols_to_drop = sorted(set(id_like_cols + high_null_cols + constant_cols))
    if cols_to_drop:
        logger.info("Dropping columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe after cleaning.")

    text_col = _detect_text_column(df, target_col)

    feature_cols = [c for c in df.columns if c != target_col]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if df[c].dtype == "object" and c != text_col]

    logger.info("Numeric columns: %s", numeric_cols)
    logger.info("Categorical columns: %s", categorical_cols)
    logger.info("Text column: %s", text_col)

    return {
        "df": df,
        "text_col": text_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_cols": cols_to_drop,
    }


def build_preprocessing_pipeline(
    df: pd.DataFrame, text_col: Optional[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    logger.info("Building preprocessing pipeline")

    analysis = analyze_schema(df, target_col)
    df = analysis["df"]
    text_col = analysis["text_col"] if text_col is None else text_col
    numeric_cols: List[str] = analysis["numeric_cols"]
    categorical_cols: List[str] = analysis["categorical_cols"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if not pd.api.types.is_numeric_dtype(y):
        logger.info("Encoding non-numeric target column: %s", target_col)
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        low_cardinality = [c for c in categorical_cols if X[c].nunique() <= 20]
        high_cardinality = [c for c in categorical_cols if X[c].nunique() > 20]

        if low_cardinality:
            cat_ohe_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat_ohe", cat_ohe_pipeline, low_cardinality))
        if high_cardinality:
            logger.info("High-cardinality categorical columns will be label-encoded at inference.")
            cat_label_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            transformers.append(("cat_label", cat_label_pipeline, high_cardinality))
    if text_col and text_col in X.columns:
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Including text column '%s' with TF-IDF", text_col)
        text_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english" if len(df) > 500 else None,
                    ),
                )
            ]
        )
        transformers.append(("text", text_pipeline, text_col))

    if not transformers:
        raise ValueError("No valid features found for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers)
    logger.info("Fitting preprocessing pipeline")
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def _wrap_logistic(preprocessor: ColumnTransformer) -> Pipeline:
    numeric_transformers = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            num_pipeline = Pipeline(
                steps=list(trans.steps) + [("scaler", StandardScaler())]
            )
            numeric_transformers.append((name, num_pipeline, cols))
        else:
            numeric_transformers.append((name, trans, cols))

    log_preprocessor = ColumnTransformer(transformers=numeric_transformers)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    return Pipeline(steps=[("preprocessor", log_preprocessor), ("model", clf)])


def _wrap_tree_model(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_select_model(
    df: pd.DataFrame, text_col: Optional[str], target_col: str
) -> Dict[str, Any]:
    logger.info("Starting training process")
    analysis = analyze_schema(df.copy(), target_col)
    df_clean = analysis["df"]
    text_col = analysis["text_col"] if text_col is None else text_col

    X = df_clean.drop(columns=[target_col])
    y_raw = df_clean[target_col]

    # Ensure target is encoded to consecutive integer classes starting at 0
    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        logger.info("Encoded non-numeric target into %d classes.", len(le.classes_))
    else:
        y_values = y_raw.values
        unique_vals = np.unique(y_values)
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        y = np.array([mapping[v] for v in y_values], dtype=int)
        logger.info("Remapped numeric target values %s to consecutive classes %s.", list(unique_vals), list(sorted(set(y))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc, y_train_proc, preprocessor = build_preprocessing_pipeline(
        df_clean, text_col, target_col
    )

    models = {
        "logistic_regression": _wrap_logistic(preprocessor),
        "random_forest": _wrap_tree_model(
            preprocessor,
            RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
            ),
        ),
        "xgboost": _wrap_tree_model(
            preprocessor,
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            ),
        ),
    }
    results: Dict[str, Dict[str, Any]] = {}
    best_model_name = None
    best_accuracy = -1.0
    best_pipeline: Optional[Pipeline] = None

    for name, pipeline in models.items():
        logger.info("Training model: %s", name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("%s accuracy: %.4f", name, acc)
        results[name] = {
            "accuracy": acc,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_model_name is None:
        raise RuntimeError("Failed to train any model.")

    ensure_project_dirs()
    logger.info("Saving best model: %s", best_model_name)
    dump(best_pipeline, MODEL_PATH)

    logger.info("Saving standalone preprocessor")
    dump(preprocessor, PREPROCESSOR_PATH)

    metrics_payload = {
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def load_metrics() -> Optional[Dict[str, Any]]:
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load metrics: %s", exc)
        return None

