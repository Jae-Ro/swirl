import json
import os
from typing import Any, Dict

from litellm import ModelResponse
from redis.asyncio import Redis

from dq_swirl.agents.orchestrator import DQAgentOrchestrator
from dq_swirl.clients.async_llm_client import AsyncLLMClient, LLMConfig
from dq_swirl.tasks.schemas import ChatTaskPayload
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


async def run_dq_agent_task(ctx: Dict[str, Any], data: Dict[str, Any]):
    """_summary_

    :param ctx: _description_
    """
    redis = Redis(connection_pool=ctx["redis_pool"])
    logger.debug(f"Task Input Data: {data}")

    req = ChatTaskPayload(**data)

    config = LLMConfig()

    llm_client = AsyncLLMClient(
        config=config,
    )
    user_query = req.prompt
    try:
        ## get messages from (user_id, conversation_id)

        ## do work
        embedding_model = ctx["embedding_model"]

        orchestrator = DQAgentOrchestrator(
            client=llm_client,
            redis=redis,
            embedding_model=embedding_model,
        )

        # TODO: make this a dataclass
        request_config = {
            "url": "http://customer:5001/api/orders",
            "method": "GET",
            "request_body": None,
        }

        async for chunk in orchestrator.run(
            request_config,
            user_query,
            data_key="raw_orders",
        ):
            await redis.publish(req.pubsub_stream_id, chunk)

    except Exception as e:
        logger.exception(e)
        error_info = json.dumps({"error": str(e), "type": "WorkerError"})
        await redis.publish(req.pubsub_stream_id, f"[ERROR]{error_info}")

    finally:
        # done
        await redis.publish(req.pubsub_stream_id, "[DONE]")
        logger.info(f"Completed DataQueryAgent Task for stream: {req.pubsub_stream_id}")

    return
