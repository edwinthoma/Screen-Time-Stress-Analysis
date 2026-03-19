import logging
from typing import Optional

import pandas as pd

from config import TARGET_COLUMN, ensure_project_dirs
from utils import load_dataset, train_and_select_model


logger = logging.getLogger("ml_system.train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(text_column: Optional[str] = None) -> None:
    ensure_project_dirs()
    try:
        df: pd.DataFrame = load_dataset()
    except FileNotFoundError as exc:
        logger.error("Training aborted: %s", exc)
        return
    except Exception as exc:
        logger.exception("Unexpected error while loading dataset: %s", exc)
        return

    try:
        metrics = train_and_select_model(df, text_column, TARGET_COLUMN)
        logger.info("Training completed. Best model: %s", metrics.get("best_model"))
        logger.info("Best accuracy: %.4f", metrics.get("best_accuracy", 0.0))
    except Exception as exc:
        logger.exception("Training failed: %s", exc)


if __name__ == "__main__":
    main()

