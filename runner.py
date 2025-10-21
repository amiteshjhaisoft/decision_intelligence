# runner.py
from typing import Dict, Any, List
from stores import load_connections, load_pipelines
from embeddings import embed_texts
from plugins.registry import SOURCES, SINKS
from secrets import resolve_secrets  # <- secret resolver (env/keyring)

BATCH = 64
DEFAULT_DIM = 384  # MiniLM dim; override via pipeline["vector_dim"] if needed


def run_pipeline_dict(p: Dict[str, Any]) -> str:
    """
    Expected pipeline dict fields:
      - name (optional)
      - source_connector, source_profile
      - destination_connector, destination_profile
      - collection (default: "Documents")
      - embedding_model (default: MiniLM)
      - embedding_normalize (default: True)
      - vector_dim (default: 384)
      - max_docs (default: 0 => all)
    """
    # --- resolve profiles ---
    conns = load_connections()
    src_id, src_prof = p["source_connector"], p["source_profile"]
    dst_id, dst_prof = p["destination_connector"], p["destination_profile"]

    if src_id not in conns or src_prof not in conns[src_id]:
        raise RuntimeError(f"Source profile not found: {src_id}.{src_prof}")
    if dst_id not in conns or dst_prof not in conns[dst_id]:
        raise RuntimeError(f"Destination profile not found: {dst_id}.{dst_prof}")

    # 🔐 resolve secrets for both ends
    src_cfg = resolve_secrets(conns[src_id][src_prof])
    dst_cfg = resolve_secrets(conns[dst_id][dst_prof])

    # --- instantiate plugins ---
    if src_id not in SOURCES:
        raise RuntimeError(f"Unknown source connector id '{src_id}'. Did you register the plugin?")
    if dst_id not in SINKS:
        raise RuntimeError(f"Unknown sink connector id '{dst_id}'. Did you register the plugin?")

    source = SOURCES[src_id](src_cfg)
    sink = SINKS[dst_id](dst_cfg)

    # --- pipeline params ---
    coll   = p.get("collection") or "Documents"
    model  = p.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    norm   = bool(p.get("embedding_normalize", True))
    dim    = int(p.get("vector_dim", DEFAULT_DIM))
    max_d  = int(p.get("max_docs") or 0)

    # --- ensure destination exists ---
    sink.ensure_destination(coll, dim)

    # --- stream -> batch -> embed -> upsert ---
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    total = 0

    for row in source.rows(max_docs=max_d):
        t = (row.get("text") or "").strip()
        if not t:
            continue
        texts.append(t)
        metas.append({k: v for k, v in row.items() if k != "text"})

        if len(texts) >= BATCH:
            vecs = embed_texts(texts, model_name=model, normalize=norm)
            sink.upsert(texts, metas, vecs)
            total += len(texts)
            texts, metas = [], []

    # flush remainder
    if texts:
        vecs = embed_texts(texts, model_name=model, normalize=norm)
        sink.upsert(texts, metas, vecs)
        total += len(texts)

    sink.close()
    return f"Inserted {total} objects into '{coll}' via {dst_id}."


def run_pipeline_by_id(pipeline_id: str) -> str:
    """Load a pipeline from pipelines.json and run it."""
    pipes = load_pipelines()
    if pipeline_id not in pipes:
        raise KeyError(f"Pipeline id not found: {pipeline_id}")
    return run_pipeline_dict(pipes[pipeline_id])
