from typing import Any, Dict, List, Optional, Tuple, TypedDict

import redis.asyncio as redis
from pydantic import BaseModel

GET_SIBLINGS_LUA = """
local meta_raw = redis.call('HGET', KEYS[1], ARGV[1])
if not meta_raw then return {} end

local meta = cjson.decode(meta_raw)
local struct_id = meta['structure_cluster_id']
local cluster_key = ARGV[2] .. struct_id

local all_members = redis.call('SMEMBERS', cluster_key)
local results = {}

for _, val in ipairs(all_members) do
    if val ~= ARGV[1] then
        table.insert(results, val)
    end
end
return results
"""


class ETLMap(TypedDict):
    semantic_cluster_id: str
    structure_cluster_id: str
    base_model_fpath: str
    parser_fpath: str
    fields: List[str]


class SignatureMetadata(BaseModel):
    semantic_cluster_id: str
    structure_cluster_id: str
    base_model_fpath: str
    parser_fpath: str
    fields: List[str]


class SignatureRegistry:
    def __init__(
        self,
        redis_url: Optional[str] = None,
        namespace: str = "etl",
    ) -> None:
        """_summary_

        :param redis_url: _description_, defaults to None
        :param namespace: _description_, defaults to "etl"
        """
        # no integration
        self.ns = namespace
        self.meta_key = f"{self.ns}:signatures:meta"
        self.cluster_prefix = f"{self.ns}:cluster:"

        # yes integration
        # TODO: refactor as this is not ideal
        if redis_url:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self._lua_get_siblings = self.redis.register_script(GET_SIBLINGS_LUA)
        else:
            self.redis = None
            self._lua_get_siblings = None

    def create_etl_lookup(
        self,
        export_map: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, ETLMap]]:
        """_summary_

        :param export_map: _description_
        :return: _description_
        """
        metadata_map = {}
        cluster_sets = {}

        for sem_id, cluster_dict in export_map.items():
            base_model_fpath = cluster_dict["base_model_fpath"]

            for struct_cluster in cluster_dict["structure_clusters"]:
                struct_id = struct_cluster["id"]
                parser_fpath = struct_cluster["parser_fpath"]
                records = struct_cluster["struct_records"]

                all_signs = [rec["signature_hash"] for rec in records]
                cluster_sets[struct_id] = all_signs

                for struct_dict in records:
                    sign = struct_dict["signature_hash"]
                    metadata_map[sign] = {
                        "semantic_cluster_id": sem_id,
                        "structure_cluster_id": struct_id,
                        "base_model_fpath": base_model_fpath,
                        "parser_fpath": parser_fpath,
                        "fields": struct_dict["fields"],
                    }

        return metadata_map, cluster_sets

    async def store_etl_lookup(
        self,
        etl_lookup_map: Dict[str, ETLMap],
        clusters: Dict[str, List[str]],
        ttl_seconds: int = 86400,
    ) -> None:
        """_summary_

        :param etl_lookup_map: _description_
        :param clusters: _description_
        :param ttl_seconds: _description_, defaults to 86400
        :return: _description_
        """
        if not self.redis:
            raise RuntimeError("No redis connection!")

        metadata_to_store = {}
        clusters = {}

        # validation
        for sign, metadata_dict in etl_lookup_map.items():
            meta = SignatureMetadata(
                semantic_cluster_id=metadata_dict["semantic_cluster_id"],
                structure_cluster_id=metadata_dict["structure_cluster_id"],
                base_model_fpath=metadata_dict["base_model_fpath"],
                parser_fpath=metadata_dict["parser_fpath"],
                fields=metadata_dict["fields"],
            )
            metadata_to_store[sign] = meta.model_dump_json()

        # write with pipeline transaction
        async with self.redis.pipeline(transaction=True) as pipe:
            if len(metadata_to_store) > 0:
                pipe.hset(
                    self.meta_key,
                    mapping=metadata_to_store,
                )
                pipe.expire(self.meta_key, ttl_seconds)

            for sig, hashes in clusters.items():
                cluster_key = f"{self.cluster_prefix}{sig}"
                # remove old sets if exist
                pipe.delete(cluster_key)
                # add hash signatures
                pipe.sadd(cluster_key, *hashes)
                pipe.expire(cluster_key, ttl_seconds)

            await pipe.execute()

        return len(metadata_to_store)

    async def lookup_hash_signature(
        self,
        signature_hash: str,
    ) -> Optional[SignatureMetadata]:
        """_summary_

        :param signature_hash: _description_
        :return: _description_
        """
        raw = await self.redis.hget(self.meta_key, signature_hash)
        return SignatureMetadata.model_validate_json(raw) if raw else None

    async def get_similar_signatures(self, signature_hash: str) -> List[str]:
        """_summary_

        :param signature_hash: _description_
        :return: _description_
        """
        return await self._lua_get_siblings(
            keys=[self.meta_key],
            args=[signature_hash, self.cluster_prefix],
        )

    async def close(self) -> None:
        """_summary_"""
        await self.redis.aclose()
