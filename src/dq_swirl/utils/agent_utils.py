import re


def extract_python_code(text: str) -> str:
    """Helper function to extract python code block from a string.

    :param text: input text string
    :return: python code string
    """
    block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(block_pattern, text, re.DOTALL)

    return match.group(1).strip() if match else ""
