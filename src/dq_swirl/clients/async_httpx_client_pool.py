import httpx

from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


async def log_response_info(response: httpx.Response) -> None:
    """Async function to log the client connection id as a callback event hook for connection pool.

    :param response: The HTTPX Response object intercepted by the event hook.
    """
    extensions = response.extensions

    # Get the actual network stream object
    stream = extensions.get("network_stream")
    # Use the object's memory ID as a unique identifier for the connection
    conn_id = id(stream) if stream else "NoStream"

    # We'll use a local cache (or just look at the logs) to see if the ID repeats
    http_version = extensions.get("http_version", b"unknown").decode()
    logger.debug(
        f"[HTTPX Pool] ConnID: {conn_id} | {http_version} | {response.request.method} {response.url}"
    )
    return


async def create_async_httpx_client_pool(
    max_connections: int = 20,
    max_keepalive_connections: int = 10,
    timeout_connect: float = 5.0,
    timeout_read: float = 10.0,
) -> httpx.AsyncClient:
    """Async function to create a pre-configured HTTPX async client pool with event hooks.

    :param max_connections: max number of concurrent total connections allowed, defaults to 20
    :param max_keepalive_connections: max number of idle connections to keep "hot" in the pool for reuse, defaults to 10
    :param timeout_connect: max time (seconds) to wait for a successful TCP/TLS handshake, defaults to 5.0
    :param timeout_read: max time (seconds) to wait for a chunk of data from the server, defaults to 10.0
    :return: configured instance of httpx.AsyncClient instance ready for use in async tasks as a connection pool
    """
    pool = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        timeout=httpx.Timeout(
            60.0,
            connect=timeout_connect,
            read=timeout_read,
        ),
        event_hooks={
            "response": [log_response_info],
        },
    )
    return pool
