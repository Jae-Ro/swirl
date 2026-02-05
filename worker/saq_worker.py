import os
import shutil
from typing import Any, Dict

import litellm
from dotenv import load_dotenv
from redis.asyncio import ConnectionPool, Redis
from saq import CronJob, Queue
from sentence_transformers import SentenceTransformer

from dq_swirl.clients.async_httpx_client import create_async_httpx_client_pool
from dq_swirl.tasks.agent_tasks import run_dq_agent_task
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()

load_dotenv("secrets.env")
load_dotenv(".env")

redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
redis_pw = os.getenv("REDIS_PW")
redis_url = f"redis://:{redis_pw}@{redis_host}:{redis_port}"


queue = Queue.from_url(redis_url, name="ai-queue")


async def cron(ctx: Dict[str, Any]):
    logger.debug("Running cron job health check")


async def startup(ctx: Dict[str, Any]):
    logger.debug("[Startup] Opening connection pools")

    # httpx pool
    pool = await create_async_httpx_client_pool()

    # set global litellm settings
    litellm.aclient_session = pool
    litellm.num_retries = 5

    # add httpx pool to context dict
    ctx["httpx_pool"] = pool

    # redis pool
    redis_pool = ConnectionPool.from_url(
        redis_url,
        max_connections=20,
        decode_responses=True,
    )
    # add redis pool to context dict
    ctx["redis_pool"] = redis_pool

    # sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./.models")
    ctx["embedding_model"] = model


async def shutdown(ctx: Dict[str, Any]):
    logger.debug("[Shutdown] Closing connection pools")
    if ctx["httpx_pool"]:
        await ctx["httpx_pool"].aclose()
    if ctx["redis_pool"]:
        await ctx["redis_pool"].aclose()

    # delete cache
    cwd = os.getcwd()
    data_cache = os.path.join(cwd, "data")
    shutil.rmtree(data_cache, ignore_errors=True)


async def before_process(ctx: Dict[str, Any]):
    logger.debug(ctx["job"])


async def after_process(ctx: Dict[str, Any]):
    pass


settings = {
    "queue": queue,
    "functions": [run_dq_agent_task],
    "concurrency": 10,
    "cron_jobs": [CronJob(cron, cron="* * * * * */60")],  # run every 1 min
    "startup": startup,
    "shutdown": shutdown,
    "before_process": before_process,
    "after_process": after_process,
}
