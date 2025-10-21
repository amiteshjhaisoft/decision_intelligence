from typing import Dict, Any, List
from stores import load_connections, load_pipelines
from embeddings import embed_texts
from plugins.registry import SOURCES, SINKS

BATCH = 64
DEFAULT_DIM = 384

def run_pipeline_dict(p: Dict[str, Any]) -> str:
    conns = load_connections()
    src_id, src_prof = p["source_connector"], p["source_profile"]
    dst_id, dst_prof = p["destination_connector"], p["destination_profile"]
    src_cfg = conns[src_id][src_prof]
    dst_cfg = conns[dst_id][dst_prof]

    source = SOURCES[src_id](src_cfg)
    sink = SINKS[dst_id](dst_cfg)

    coll = p.get("collection") or "Documents"
    model = p.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    dim = int(p.get("vector_dim", DEFAULT_DIM))
    max_d = int(p.get("max_docs") or 0)

    sink.ensure_destination(coll, dim)

    texts, metas, total = [], [], 0
    for row in source.rows(max_docs=max_d):
        t = (row.get("text") or "").strip()
        if not t: continue
        texts.append(t)
        metas.append({k:v for k,v in row.items() if k != "text"})
        if len(texts) >= BATCH:
            vecs = embed_texts(texts, model_name=model)
            sink.upsert(texts, metas, vecs)
            total += len(texts)
            texts, metas = [], []

    if texts:
        vecs = embed_texts(texts, model_name=model)
        sink.upsert(texts, metas, vecs)
        total += len(texts)

    sink.close()
    return f"Inserted {total} objects into '{coll}' via {dst_id}."
