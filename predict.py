import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

from config import METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN, ensure_project_dirs


logger = logging.getLogger("ml_system.predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_artifacts() -> Tuple[Any, Optional[Dict[str, Any]]]:
    ensure_project_dirs()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load(MODEL_PATH)
    metrics = None
    if METRICS_PATH.exists():
        try:
            import json

            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as exc:
            logger.error("Failed to load metrics: %s", exc)
    return model, metrics


def predict_from_dict(model: Any, data: Dict[str, Any]) -> Tuple[int, float]:
    df = pd.DataFrame([data])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(df)
            if proba_arr.ndim == 2:
                proba = float(np.max(proba_arr[0]))
        except Exception as exc:
            logger.error("Failed to compute prediction probabilities: %s", exc)
    pred = model.predict(df)[0]
    confidence = float(proba) if proba is not None else 0.0
    return int(pred), confidence


def get_feature_importance(model: Any) -> Optional[Dict[str, float]]:
    pipe = model
    if not hasattr(pipe, "named_steps") or "preprocessor" not in pipe.named_steps:
        return None

    preprocessor = pipe.named_steps["preprocessor"]
    model_step = pipe.named_steps.get("model")

    if model_step is None or not hasattr(model_step, "feature_importances_"):
        return None

    try:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(cols)
                except TypeError:
                    fn = trans.get_feature_names_out()
                feature_names.extend(list(fn))
            else:
                feature_names.extend(cols if isinstance(cols, list) else [cols])

        importances = model_step.feature_importances_
        if len(feature_names) != len(importances):
            return None

        return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
    except Exception as exc:
        logger.error("Failed to extract feature importances: %s", exc)
        return None

