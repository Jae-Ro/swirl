import time
from collections import Counter
from typing import List

import pytest
from dotenv import load_dotenv

from dq_swirl.agents.etl_builder_agent import ETLBuilderAgent
from dq_swirl.clients.async_llm_client import AsyncLLMClient, LLMConfig
from dq_swirl.ingestion.rust_ingestion import smart_parse_batch
from dq_swirl.ingestion.structure_analyzer import StructuralAnalyzer
from dq_swirl.ml_ai.clustering import ClusterOrchestrator
from dq_swirl.utils.log_utils import get_custom_logger
from tests.conftest import LLM_CONFIGS

load_dotenv("secrets.env")


logger = get_custom_logger()


class TestETLBuilderAgent:
    @pytest.mark.parametrize("config", LLM_CONFIGS)
    async def test_agent_run(self, config: LLMConfig, messy_data: List[str]):
        clusterer = ClusterOrchestrator()
        client = AsyncLLMClient(config=config)
        agent = ETLBuilderAgent(client=client)

        start = time.time()
        logger.debug(f"SAMPLES: {len(messy_data)}")

        data_samples = smart_parse_batch(messy_data)
        analyzer = StructuralAnalyzer(ignore_unparsed=False)

        hash_counts = Counter()
        unique_structures = {}

        for raw, parsed in data_samples:
            result = analyzer.generate_fingerprint(raw, parsed)
            signature_hash = result["hash"]
            hash_counts[signature_hash] += 1
            unique_structures[signature_hash] = unique_structures.get(
                signature_hash, result
            )

        logger.debug(
            f"Detected {len(unique_structures)} unique schemas across {len(data_samples)} records.\n"
        )

        registry = analyzer.get_signature_map()
        cluster_map = clusterer.make_clusters(registry)
        end = time.time()
        logger.debug(
            f"PROCESSING RUNTIME: {round(end - start, 2)} seconds for {len(messy_data)} samples!"
        )

        start = end

        export_map = await agent.run(cluster_map)

        end = time.time()
        logger.debug(
            f"CODING RUNTIME: {round(end - start, 2)} seconds for {len(messy_data)} samples!"
        )
        assert len(export_map) > 1
