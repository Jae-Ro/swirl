import json
import os
from typing import Any, Dict

from litellm import ModelResponse
from redis.asyncio import Redis

from dq_swirl.clients.async_llm_client import AsyncLLMClient
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

    llm_client = AsyncLLMClient(
        model=req.model,
        api_base=os.getenv("LLM_BASE_URL"),
    )
    try:
        ## get messages from (user_id, conversation_id)

        ## do work

        ## stream final llm response
        response = await llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": req.prompt,
                },
            ],
            stream=True,
        )

        async for chunk in response:
            chunk: ModelResponse
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                await redis.publish(req.pubsub_stream_id, content)

    except Exception as e:
        logger.exception(e)
        error_info = json.dumps({"error": str(e), "type": "WorkerError"})
        await redis.publish(req.pubsub_stream_id, f"[ERROR]{error_info}")

    finally:
        # done
        await redis.publish(req.pubsub_stream_id, "[DONE]")
        logger.info(f"Completed DataQueryAgent Task for stream: {req.pubsub_stream_id}")

    return
