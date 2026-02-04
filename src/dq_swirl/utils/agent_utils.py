import asyncio
import random
import re
from functools import wraps


def extract_python_code(text: str) -> str:
    """Helper function to extract python code block from a string.

    :param text: input text string
    :return: python code string
    """
    block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(block_pattern, text, re.DOTALL)

    return match.group(1).strip() if match else ""


def extract_sql_code(text: str) -> str:
    """Helper function to extract sql code block from a string.

    :param text: input text string
    :return: sql code string
    """
    block_pattern = r"```(?:sql)?\s*(.*?)\s*```"
    match = re.search(block_pattern, text, re.DOTALL)

    return match.group(1).strip() if match else ""


def prepause(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await asyncio.sleep(random.uniform(0.2, 0.5))
        return await func(*args, **kwargs)

    return wrapper
