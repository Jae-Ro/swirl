from typing import Dict, List, TypedDict

import numpy as np
from sklearn.cluster import HDBSCAN
from transformers import logging as transformers_logging

from dq_swirl.ingestion.structure_analyzer import SignatureEntry
from dq_swirl.ml_ai.embedding_model import EmbeddingModel, load_sentence_transformer

transformers_logging.set_verbosity_error()

#################################################################################
############################## Semantic Clustering ##############################
#################################################################################


class SemanticClusterData(TypedDict):
    cluster_id: str
    fields: List[str]
    signature_hash: str
    is_outlier: bool


class SemanticClusterer:
    def __init__(
        self,
        embedding_model: str | EmbeddingModel = "all-MiniLM-L6-v2",
        model_cache_dir: str = "./.models",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        distance_metric: str = "cosine",
        cluster_selection_epsilon: float = 0.08,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = True,
    ) -> None:
        """_summary_

        :param embedding_model: _description_, defaults to "all-MiniLM-L6-v2"
        :param model_cache_dir: _description_, defaults to "./.models"
        :param min_cluster_size: _description_, defaults to 2
        :param min_samples: _description_, defaults to 1
        :param distance_metric: _description_, defaults to "cosine"
        :param cluster_selection_epsilon: _description_, defaults to 0.08
        :param cluster_selection_method: _description_, defaults to "eom"
        :param allow_single_cluster: _description_, defaults to True
        """
        self.embedding_model = embedding_model
        self.model_cache_dir = model_cache_dir
        self.min_cluster_size = min_cluster_size

        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster

        # init embedding model if not passed in
        if isinstance(embedding_model, str):
            self.embedding_model = load_sentence_transformer(self.embedding_model)

        # init clusterer
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.distance_metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            copy=True,
        )

    def fit_predict(
        self,
        registry_map: Dict[str, SignatureEntry],
    ) -> Dict[str, List[SemanticClusterData]]:
        """_summary_

        :param registry_map: _description_
        :return: _description_
        """
        hashes = list(registry_map.keys())

        signatures_as_text = []
        for h in hashes:
            h_dict = dict(registry_map[h]["signature"])
            # remove the 'black hole' field that swallows everything
            # h_dict = {k: v for k, v in h_dict.items() if k != "_unparsed"}
            h_dict.pop("_unparsed", None)

            # sort keys to ensure structural identity regardless of log order
            sorted_keys = sorted(h_dict.keys())

            if not sorted_keys:
                text_rep = "schema:empty_blob"
            else:
                # 'field:' prefix to define the role of the tokens
                text_rep = " ".join([f"field:{k}" for k in sorted_keys])

            signatures_as_text.append(text_rep)

        # run embedding
        embeddings = self.embedding_model.encode(signatures_as_text)
        X = np.ascontiguousarray(embeddings, dtype=np.float64)

        # fit_predict()
        labels = self.clusterer.fit_predict(X)

        conjoined_map = {}
        for i, cluster_id in enumerate(labels):
            h = hashes[i]
            # unique IDs to outliers so they don't group into one '-1' bucket
            final_id = int(cluster_id) if cluster_id != -1 else (400 + i)
            final_id = f"{final_id}"

            record = {
                "cluster_id": final_id,
                "fields": list(registry_map[h]["signature"].keys()),
                "signature_hash": h,
                "is_outlier": bool(cluster_id == -1),
            }
            conjoined_map[final_id] = conjoined_map.get(final_id, [])
            conjoined_map[final_id].append(record)

        return conjoined_map
