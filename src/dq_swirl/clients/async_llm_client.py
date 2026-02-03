import os
from typing import AsyncIterable, Dict, List, Optional

import litellm
from litellm import CustomStreamWrapper, ModelResponse

from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class AsyncLLMClient:
    """
    A high-level asynchronous client for interacting with Large Language Models via LiteLLM.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: Optional[str] = None,
    ) -> None:
        """Method to init the AsyncLLMClient with default model and API settings.

        :param model: default model identifier (e.g., "openai/google/gemma-3-27b-it")
        :param api_base: base URL for the LLM API provider (e.g., "http://localhost:8000/v1")
        :param api_key: optional parameter for LLM API Key
        """
        self.model = model
        self.api_base = api_base
        self._api_key = api_key

        if self._api_key is None:
            # placeholder
            self._api_key = "123"

    def __repr__(self):
        return f"AsyncLLMClient(base_url={self.api_base}, model={self.model})"

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
        model = self.model
        if model_override:
            model = model_override

        # handle api url override
        api_base = self.api_base
        if base_url_override:
            api_base = base_url_override

        # handle api key override
        api_key = self._api_key
        if api_key_override:
            api_key = api_key_override

        return await litellm.acompletion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            **lite_llm_kwargs,
        )
