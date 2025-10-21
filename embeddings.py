from typing import List
from sentence_transformers import SentenceTransformer

_cache = {}
def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize=True):
    model = _cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _cache[model_name] = model
    return model.encode(texts, normalize_embeddings=normalize)
