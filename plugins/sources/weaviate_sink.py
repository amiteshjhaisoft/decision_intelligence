from typing import Dict, Any, List
from weaviate import connect_to_wcs, connect_to_custom
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from plugins.registry import register_sink

@register_sink("weaviate")
class WeaviateSink:
    def __init__(self, cfg: Dict[str, Any]):
        api_key = cfg.get("api_key")
        auth = Auth.api_key(api_key) if api_key else None
        cluster_url = (cfg.get("cluster_url") or cfg.get("url") or "").strip().rstrip("/")
        if cluster_url:
            self.client = connect_to_wcs(cluster_url=cluster_url, auth_credentials=auth)
        else:
            scheme = (cfg.get("scheme") or "https").lower()
            host = cfg.get("host", "localhost")
            port = int(cfg.get("port", 8080))
            self.client = connect_to_custom(http_host=host, http_port=port, http_secure=(scheme=="https"), auth_credentials=auth)
        self.collection = None

    def ensure_destination(self, name: str, dim: int):
        names = {c.name for c in self.client.collections.list_all()}
        if name not in names:
            self.client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source_path", data_type=DataType.TEXT),
                ],
            )
        self.collection = self.client.collections.get(name)

    def upsert(self, texts: List[str], payloads: List[Dict[str, Any]], vectors):
        props = [{"text": t, "source_path": p.get("source_path", "")} for t,p in zip(texts, payloads)]
        self.collection.data.insert_many(properties=props, vectors=vectors)

    def close(self): self.client.close()
