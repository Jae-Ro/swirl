import os

import pytest
from dotenv import load_dotenv

from dq_swirl.clients.async_llm_client import LLMConfig

load_dotenv("secrets.env")
load_dotenv(".env")


DEFAULT_MODEL = os.getenv("LLM_MODEL")
DEFAULT_LLM_URL = os.getenv("LLM_BASE_URL")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "123")

LLM_CONFIGS = [
    pytest.param(
        LLMConfig(
            model=DEFAULT_MODEL, base_url=DEFAULT_LLM_URL, api_key=DEFAULT_LLM_API_KEY
        )
    )
]
