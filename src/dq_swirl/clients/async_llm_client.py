import os
from dataclasses import dataclass, field
from typing import AsyncIterable, Dict, List, Optional

import litellm
from litellm import ModelResponse

from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Configuration dataclass for LLM client

    :param model: default model identifier (e.g., "openai/google/gemma-3-27b-it")
    :param base_url: base URL for the LLM API provider (e.g., "http://localhost:8000/v1")
    :param api_key: optional parameter for LLM API Key
    """

    model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
        )
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
        )
    )
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY",
            "123",
        )
    )


class AsyncLLMClient:
    """
    A high-level asynchronous client for interacting with Large Language Models via LiteLLM.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """Method to init the AsyncLLMClient with default model and API settings.

        :param config: instance of LLMConfig
        """
        self.config = config

        # if not provided, try to create one from ENV vars
        if self.config is None:
            self.config = LLMConfig()

    def __repr__(self):
        return f"AsyncLLMClient({self.config})"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        **lite_llm_kwargs,
    ) -> ModelResponse | AsyncIterable[ModelResponse]:
        """Async method to send a chat completion request to configured LLM provider inference endpoint

        Supports optional overrides for model, URL, and credentials on a
        per-call basis. It handles both synchronous returns and asynchronous streaming
        iterators based on the 'stream' parameter passed in lite_llm_kwargs

        :param messages: list of message dictionaries (e.g., {"role": "user", "content": "..."})
        :param model_override: optional model name to use instead of the default, defaults to None
        :param base_url_override: optional API base URL to use instead of the default, defaults to None
        :param api_key_override: optional API key to use instead of the environment default, defaults to None
        :param lite_llm_kwargs: additional parameters passed to litellm.acompletion (e.g., stream, temperature, max_tokens)
        :return: either instance of litellm.ModelResponse or AsyncIterable where each item is of type litellm.ModelResponse
        """
        # handle model override
        model = self.config.model
        if model_override:
            model = model_override

        # handle api url override
        api_base = self.config.base_url
        if base_url_override:
            api_base = base_url_override

        # handle api key override
        api_key = self.config.api_key
        if api_key_override:
            api_key = api_key_override

        return await litellm.acompletion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            **lite_llm_kwargs,
        )
