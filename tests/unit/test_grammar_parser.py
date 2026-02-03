from typing import List

from dq_swirl.ingestion.rust_ingestion import smart_parse_batch
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class TestGrammarParser:
    def test_smart_parse_batch(self, messy_data: List[str]):
        res = smart_parse_batch(messy_data)

        for raw, parsed in res:
            logger.debug(f"RAW: {raw} | PARSED: {parsed}")
            assert isinstance(raw, str)
            assert isinstance(parsed, dict)
            assert len(parsed) > 0
