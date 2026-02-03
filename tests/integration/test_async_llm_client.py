import pytest
from litellm import ModelResponse

from dq_swirl.clients.async_llm_client import AsyncLLMClient, LLMConfig
from dq_swirl.utils.log_utils import get_custom_logger
from tests.conftest import LLM_CONFIGS

logger = get_custom_logger()


class TestAsyncLLMClient:
    @pytest.mark.parametrize("config", LLM_CONFIGS)
    async def test_hello_world_batch(self, config: LLMConfig):
        client = AsyncLLMClient(config=config)

        resp = await client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            stream=False,
            max_tokens=100,
        )

        content = resp.choices[0].message.content
        logger.debug(content)
        assert isinstance(content, str) and len(content) > 0

    @pytest.mark.parametrize("config", LLM_CONFIGS)
    async def test_hello_world_streaming(self, config: LLMConfig):
        client = AsyncLLMClient(config=config)

        resp = await client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            stream=True,
            max_tokens=100,
        )

        buffer = []
        async for chunk in resp:
            chunk: ModelResponse
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assert content
                logger.debug(content)
                buffer.append(content)

        content = "".join(buffer)
        logger.debug(content)
        assert isinstance(content, str) and len(content) > 0
