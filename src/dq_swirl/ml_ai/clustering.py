from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from dq_swirl.ingestion.structure_analyzer import SignatureEntry
from dq_swirl.ml_ai.embedding_model import EmbeddingModel
from dq_swirl.ml_ai.semantic_clustering import SemanticClusterer
from dq_swirl.ml_ai.structure_clustering import StructureClusterer


@dataclass(slots=True)
class StructureClusterParams:
    min_cluster_size: int = 2
    distance_metric: str = "euclidean"
    tfidf_analyzer: str = "char"
    tfidf_ngram_range: Tuple[int, int] = (3, 5)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SemanticClusterParams:
    model_cache_dir: str = "./.models"
    min_cluster_size: int = 2
    min_samples: int = 1
    distance_metric: str = "cosine"
    cluster_selection_epsilon: float = 0.08
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ClusterRecord:
    signature_hash: str
    semantic_cluster_id: int
    structure_cluster_id: int
    raw: str
    parsed: Dict[str, Any]
    fields: List[str]


class ClusterOrchestrator:
    def __init__(
        self,
        structure_cluster_params: StructureClusterParams = StructureClusterParams(),
        semantic_cluster_params: SemanticClusterParams = SemanticClusterParams(),
        embedding_model: str | EmbeddingModel = "all-MiniLM-L6-v2",
    ) -> None:
        """_summary_

        :param structure_cluster_params: _description_
        :param semantic_cluster_params: _description_
        """

        self.structure_clusterer = StructureClusterer(
            **structure_cluster_params.to_dict(),
        )
        self.semantic_clusterer = SemanticClusterer(
            embedding_model=embedding_model,
            **semantic_cluster_params.to_dict(),
        )

        # vars to populate
        self.structure_hash_map = None
        self.semantic_cluster_map = None
        self.cluster_map = None

    def make_clusters(
        self, registry_map: Dict[str, SignatureEntry]
    ) -> Dict[str, List[ClusterRecord]]:
        """_summary_

        :param registry_map: _description_
        :return: _description_
        """
        self.structure_hash_map = self.structure_clusterer.fit_predict(registry_map)
        self.semantic_cluster_map = self.semantic_clusterer.fit_predict(registry_map)

        cluster_dict = {}
        for cluster_id, records in self.semantic_cluster_map.items():
            cluster_dict[cluster_id] = cluster_dict.get(cluster_id, [])
            for rec in records:
                signature_hash = rec["signature_hash"]
                structure_cluster_id = self.structure_hash_map[signature_hash][
                    "cluster_id"
                ]
                analyzer_records = registry_map[signature_hash]["records"]
                fields_li = rec["fields"]
                # ignore _unparsed
                fields_li = [f for f in fields_li if f != "_unparsed"]
                if len(fields_li) < 1:
                    continue
                for r in analyzer_records:
                    parsed_dict = r["parsed"]
                    # ignore _unparsed
                    parsed_dict.pop("_unparsed", None)
                    if len(parsed_dict) < 1:
                        continue
                    cluster_dict[cluster_id].append(
                        ClusterRecord(
                            signature_hash=signature_hash,
                            semantic_cluster_id=cluster_id,
                            structure_cluster_id=structure_cluster_id,
                            raw=r["raw"],
                            parsed=parsed_dict,
                            fields=fields_li,
                        )
                    )

        return cluster_dict
