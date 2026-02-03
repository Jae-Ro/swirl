from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from dq_swirl.ingestion.structure_analyzer import SignatureEntry

#################################################################################
############################# Structural Clustering #############################
#################################################################################


@dataclass(slots=True)
class StructClusterData:
    cluster_id: int
    fields: List[str]
    signature_hash: str
    is_outlier: bool


class StructureClusterer:
    def __init__(
        self,
        min_cluster_size: int = 2,
        distance_metric: str = "euclidean",
        tfidf_analyzer: str = "char",
        tfidf_ngram_range: Tuple[int, int] = (3, 5),
    ) -> None:
        """_summary_

        :param min_cluster_size: _description_, defaults to 2
        :param distance_metric: _description_, defaults to "euclidean"
        :param tfidf_analyzer: _description_, defaults to "char"
        :param tfidf_ngram_range: _description_, defaults to (3, 5)
        """
        self.min_cluster_size = min_cluster_size
        self.distance_metric = distance_metric
        self.tfidf_analyzer = tfidf_analyzer
        self.tfidf_ngram_range = tfidf_ngram_range

        # init vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer=self.tfidf_analyzer, ngram_range=self.tfidf_ngram_range
        )

        # init clusterer
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.distance_metric,
            copy=True,
            prediction_data=True,
        )

    def _to_text(self, signature_dict: Dict[str, Any]) -> str:
        """Utility method to convert a single signature dict to a space-separated string of keys

        :param signature_dict: input signature structure dict
        :return: string representation of structure
        """
        return " ".join(signature_dict.keys())

    def fit_predict(
        self,
        registry_map: Dict[str, SignatureEntry],
    ) -> Dict[str, StructClusterData]:
        """_summary_

        :param registry_output: _description_
        :return: _description_
        """
        hashes = list(registry_map.keys())
        signatures_as_text = [
            self._to_text(val.signature) for val in registry_map.values()
        ]

        matrix = self.vectorizer.fit_transform(signatures_as_text)

        # fit_predict()
        labels = self.clusterer.fit_predict(matrix.toarray())

        conjoined_map = {}
        for i, cluster_id in enumerate(labels):
            h = hashes[i]
            conjoined_map[h] = StructClusterData(
                cluster_id=int(cluster_id),
                fields=list(registry_map[h].signature.keys()),
                signature_hash=h,
                is_outlier=bool(cluster_id == -1),
            )

        return conjoined_map
