import json
from typing import List

from dq_swirl.ingestion.rust_ingestion import smart_parse_batch
from dq_swirl.ingestion.structure_analyzer import StructuralAnalyzer
from dq_swirl.ml_ai.clustering import ClusterOrchestrator
from dq_swirl.utils.dataclass_utils import DataclassEncoder
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class TestClusterOrchestrator:
    async def test_string_clustering(self, messy_data: List[str]):
        # grammar parsing
        logger.debug("Running Grammar Parsing")
        samples = smart_parse_batch(messy_data)
        assert len(samples) == len(messy_data)
        assert len(samples[0]) == 2

        # structural analyzer
        logger.debug("Running Structure Analyzer")
        analyzer = StructuralAnalyzer()
        for raw, parsed in samples:
            analyzer.generate_fingerprint(raw, parsed, store_in_map=True)

        signature_map = analyzer.get_signature_map()
        assert len(signature_map) > 0

        # clustering
        logger.debug("Running Clustering")
        cluster_op = ClusterOrchestrator()
        cluster_map = cluster_op.make_clusters(signature_map)
        cluster_map_dump = json.dumps(cluster_map, cls=DataclassEncoder, indent=4)
        logger.debug(cluster_map_dump)
