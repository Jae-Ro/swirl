import os
from typing import List

import litellm
import pytest
from dotenv import load_dotenv

from dq_swirl.clients.async_llm_client import LLMConfig

load_dotenv("secrets.env")
load_dotenv(".env")

litellm.num_retries = 5

DEFAULT_MODEL = os.getenv("LLM_MODEL")
DEFAULT_LLM_URL = os.getenv("LLM_BASE_URL")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "123")

LLM_CONFIGS = [
    pytest.param(
        LLMConfig(
            model=DEFAULT_MODEL,
            base_url=DEFAULT_LLM_URL,
            api_key=DEFAULT_LLM_API_KEY,
        )
    )
]

MESSY_SAMPLE_DATA = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1004:   Buyer=  AMANDA SMITH ,Location=Seattle, WA,Total=$50.00, Items: desk lamp",
    "Order 1005: Buyer=Raj Patel, Total=1,200.50, Items: monitor, stand, cable",
    "Order 1006: total=$89.99, location=Miami, FL, buyer=Elena Rossi, Items: keyboard",
    "Order 1007: Buyer=Chris P., Location=Denver, CO, Total=$12.00, Items: stickers -- [DISCOUNT APPLIED]",
    "Order 1008: Buyer=O'Connor, S., Location=Portland, OR, Total=$0.00, Items: ",
    "Order 1011: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1012: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1013: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
    "Order 1014: Buyer=Rachel Kim, Locadtion=Seattle, WA, Total=$89.50, Items: coffee maker",
    "Order 1015: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp",
    "Order=1016, Buyer=Jake Myers, Total=$1,512.00, Items: monitor,",
    '{"id": "usr_001", "name": "Alex Johnson", "role": "admin", "isActive": true, "createdAt": "2025-11-02T09:14:23Z"}',
    '{"id": "usr_002", "name": "Maria Lopez", "email": "maria.lopez@example.com", "role": "editor", "isActive": null, "createdAt": "2025-12-18T16:47:10Z", "lastLoginIp": "192.168.1.42"}',
    '{"id": "usr_003", "email": "samir.patel@example.com", "role": "viewer", "isActive": false, "createdAt": "08/05/2024"}',
    '{"id": 4, "name": "Chen Wei", "email": "chen.wei@example.com", "isActive": true, "createdAt": null}',
    '{"id": "usr_005", "name": "Broken Record", "email": "broken@example.com"}',
]


@pytest.fixture(scope="class")
def messy_data() -> List[str]:
    return MESSY_SAMPLE_DATA
