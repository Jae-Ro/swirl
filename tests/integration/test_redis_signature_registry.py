from dq_swirl.persistence.signature_registry import SignatureMetadata, SignatureRegistry
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class TestRedisSignatureRegistry:
    async def test_signature_registry_create(
        self,
        redis_url,
        etl_lookup_map,
        cluster_sets,
    ) -> None:
        """_summary_

        :param redis_url: _description_
        :param etl_lookup_map: _description_
        :param cluster_sets: _description_
        """
        registry = SignatureRegistry(redis_url=redis_url)
        # store them
        await registry.store_etl_lookup(etl_lookup_map, cluster_sets)
        # read and validate
        cand_hash = "50eb97a85647221ecc7f65f74d68d156"
        res = await registry.lookup_hash_signature(cand_hash)
        assert isinstance(res, SignatureMetadata)
        logger.debug(res.model_dump_json(indent=4))
        # get similar hashes
        similar = await registry.get_similar_signatures(cand_hash)
        logger.debug(f"Similar Hashes: {similar}")
        assert len(similar) > 0
        assert cand_hash not in similar
