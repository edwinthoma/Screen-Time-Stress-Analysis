import os
import logging
from typing import Any, Dict

import requests


logger = logging.getLogger("ml_system.hf_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


API_URL = "https://router.huggingface.co/v1/chat/completions"


def _get_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return {"Authorization": f"Bearer {token}"}


def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = _get_headers()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def generate_ai_response(ticket_text: str) -> str:
    payload: Dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional customer support assistant.",
            },
            {"role": "user", "content": ticket_text},
        ],
        "model": "deepseek-ai/DeepSeek-R1:novita",
    }
    try:
        result = query(payload)
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        return f"LLM Error: {str(exc)}"

