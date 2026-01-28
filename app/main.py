import dataclasses
import json
import os
import uuid
import virt_s3
from typing import Annotated, AsyncGenerator, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel
from quart import Quart, Response, make_response, request, stream_with_context
from quart_cors import cors
from redis.asyncio import from_url
from saq import Queue

from dq_swirl.tasks.schemas import ChatTaskPayload, CleanupTaskPayload
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()

load_dotenv("secrets.env")
load_dotenv(".env")

app = Quart(__name__)
app = cors(app, allow_origin="*")

# global connections
redis_client = None
task_queue = None


class ChatRequest(BaseModel):
    """
    BaseModel class to dictate chat request parameters.

    Attributes:
        prompt: user's input message -- must not be empty string (required)
        model: name of model to be run (required)
        user_id: id of user (required)
        conversation_id: id of chat conversation from provided user (required)
    """

    # for converting camelCase to snake_case
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
    # attributes
    prompt: str = Annotated[str, Field(min_length=1)]
    model: str
    user_id: str
    conversation_id: str


@app.before_serving
async def setup_connections():
    """Initialize Redis and SAQ Queue before the app starts serving."""
    global redis_client, task_queue
    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")
    redis_pw = os.getenv("REDIS_PW")
    redis_url = f"redis://:{redis_pw}@{redis_host}:{redis_port}"

    # init the primary Redis client for Pub/Sub and SAQ
    redis_client = from_url(redis_url, decode_responses=False)
    task_queue = Queue.from_url(redis_url, name="ai-queue")

    logger.info("Successfully connected to Redis and SAQ Queue.")

    # create s3 bucket
    params = virt_s3.get_default_params()
    with virt_s3.SessionManager(params=params) as session:
        virt_s3.create_bucket("app-local", params=params, client=session)


@app.after_serving
async def close_connections():
    """Gracefully close connections on shutdown."""
    if redis_client:
        await redis_client.close()
        logger.info("Closed Redis connections.")


@app.route("/health", methods=["GET"])
async def health_check() -> Response:
    return await make_response({"status": "ok"}, 200)


@app.route("/api/chat", methods=["POST"])
async def chat() -> Response:
    """Function to handle /api/chat route

    :return: Response with async generator for LLM/Agent request call
    :yield: SSE response chunk string
    """
    body = await request.get_json()
    logger.info(f"Handling Chat Request:\n{json.dumps(body, indent=4)}")

    # handling request body validation
    try:
        body = ChatRequest.model_validate(body)
    except ValidationError as e:
        logger.exception(e)
        return await make_response({"error": e.errors()}, 400)

    # create pubsub streamid
    stream_id = str(uuid.uuid4())

    # create task payload
    payload = ChatTaskPayload(
        user_id=body.user_id,
        conversation_id=body.conversation_id,
        model=body.model,
        prompt=body.prompt,
        pubsub_stream_id=stream_id,
    )

    # redis queue
    await task_queue.enqueue(
        "run_dq_agent_task",
        data=dataclasses.asdict(payload),
        timeout=300,
    )

    @stream_with_context
    async def generate() -> AsyncGenerator[str, None]:
        # initial heartbeat to establish the connection
        yield ": heartbeat\n\n"

        async with redis_client.pubsub() as pubsub:
            logger.info(f"Subscribed to {stream_id}")
            await pubsub.subscribe(stream_id)
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        content = message["data"].decode("utf-8")

                        # end of stream
                        if content == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break

                        # error in stream
                        if content.startswith("[ERROR]"):
                            raise Exception(content)

                        # relay the event
                        chunk_payload = json.dumps(
                            {
                                "data": {
                                    "content": content,
                                }
                            }
                        )
                        yield f"data: {chunk_payload}\n\n"

            except Exception as e:
                error_msg = f"{e}".replace("[ERROR]", "")
                error_payload = json.dumps(
                    {
                        "error": error_msg,
                    }
                )
                yield f"data: {error_payload}"
                yield "data: [DONE]\n\n"

            finally:
                await pubsub.unsubscribe(stream_id)
                logger.info(f"Unsubscribed from {stream_id}")

    # create response stream object
    response = await make_response(generate())

    # set headers
    response.mimetype = "text/event-stream"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Transfer-Encoding"] = "chunked"

    # set timeout to be 5 min
    response.timeout = 300

    return response


if __name__ == "__main__":
    app.run(
        port=5000,
        debug=True,
        keep_alive_timeout=300,
    )
