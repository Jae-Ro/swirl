from typing import List

import pytest
from redis.asyncio import Redis

from dq_swirl.agents.orchestrator import DQAgentOrchestrator
from dq_swirl.clients.async_httpx_client import AsyncHttpxClient
from dq_swirl.clients.async_llm_client import AsyncLLMClient, LLMConfig
from dq_swirl.utils.log_utils import get_custom_logger
from tests.conftest import LLM_CONFIGS

logger = get_custom_logger()


class TestDQAgentOrchestrator:
    @pytest.mark.parametrize("config", LLM_CONFIGS)
    async def test_agent_orchestrator(
        self,
        config: LLMConfig,
        redis_client: Redis,
    ):
        llm_client = AsyncLLMClient(config=config)
        agent = DQAgentOrchestrator(
            client=llm_client,
            redis=redis_client,
        )

        request_config = {
            "url": "http://localhost:5001/api/orders",
            "method": "GET",
            "request_body": None,
        }
        query = "Show me all orders where the buyer was located in Ohio and total value was over 500"
        await agent.run(
            request_config,
            user_query=query,
            data_key="raw_orders",
        )
