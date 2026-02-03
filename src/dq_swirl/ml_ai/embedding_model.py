from typing import Protocol


class EmbeddingModel(Protocol):
    """Duck type protocol class for anything that has a `.encode()` method

    :param Protocol: extends typing.Protocol
    """

    def encode(self, *args, **kwargs):
        pass


def load_sentence_transformer(model_name: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
