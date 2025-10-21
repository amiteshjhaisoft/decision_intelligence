# runner.py
from __future__ import annotations
import os, io, re, json, time, hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

from stores import load_connections, load_pipelines

# ---------- Optional imports (keep graceful) ----------
try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None
try:
    import weaviate
    from weaviate import connect_to_wcs, connect_to_custom
    from weaviate.classes.config import Property, DataType, Configure
    from weaviate.classes.init import Auth
except Exception:
    weaviate = None

# Chunking / Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------- Utilities ----------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()

def _iter_local_files(root: str, pattern: str = r".*\.(txt|md|csv|json|pdf|docx|xlsx)$") -> Iterable[Tuple[str, bytes]]:
    rootp = Path(root)
    rx = re.compile(pattern, re.I)
    for p in rootp.rglob("*"):
        if p.is_file() and rx.match(p.name):
            try: yield (str(p), p.read_bytes())
            except Exception: pass

def _iter_azblob_files(cfg: Dict[str, Any], container: str, prefix: str = "", pattern: str = r".*") -> Iterable[Tuple[str, bytes]]:
    if BlobServiceClient is None:
        raise RuntimeError("Install azure-storage-blob for Azure Blob source.")
    # auth paths: connection_string OR (account_name + account_key) OR sas_url
    if cfg.get("connection_string"):
        svc = BlobServiceClient.from_connection_string(cfg["connection_string"])
    elif cfg.get("sas_url"):
        account_url = cfg["sas_url"].split("?")[0]
        svc = BlobServiceClient(account_url=account_url, credential=cfg["sas_url"])
    elif cfg.get("account_name") and cfg.get("account_key"):
        url = f"https://{cfg['account_name']}.blob.core.windows.net"
        svc = BlobServiceClient(account_url=url, credential=cfg["account_key"])
    else:
        raise RuntimeError("AzureBlob profile must provide connection_string or sas_url or (account_name+account_key)")

    rx = re.compile(pattern, re.I)
    cont = svc.get_container_client(container)
    for blob in cont.list_blobs(name_starts_with=prefix or ""):
        if rx.match(Path(blob.name).name):
            data = cont.download_blob(blob.name).readall()
            yield (f"az://{container}/{blob.name}", data)

def _bytes_to_text(name: str, data: bytes) -> Optional[str]:
    ext = Path(name).suffix.lower()
    try:
        if ext in {".txt", ".md", ".csv", ".json"}:
            return data.decode("utf-8", "ignore")
        if ext == ".pdf":
            import pypdf
            txt = []
            with io.BytesIO(data) as f:
                r = pypdf.PdfReader(f)
                for page in r.pages:
                    try: txt.append(page.extract_text() or "")
                    except Exception: pass
            return "\n".join(txt).strip()
        if ext == ".docx":
            import docx2txt, tempfile
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
                tmp.write(data); tmp.flush()
                return docx2txt.process(tmp.name) or ""
        # (xlsx) – light extractor: join visible cell values
        if ext in {".xlsx", ".xls"}:
            import pandas as pd
            with io.BytesIO(data) as f:
                xls = pd.read_excel(f, sheet_name=None, dtype=str)
            return "\n\n".join(
                f"[Sheet: {sn}]\n" + df.fillna("").to_csv(index=False)
                for sn, df in xls.items()
            )
    except Exception:
        return None
    return None

def _make_splitter(cfg: Dict[str, Any]):
    method = (cfg.get("method") or "recursive").lower()
    chunk_size = int(cfg.get("chunk_size") or 800)
    chunk_overlap = int(cfg.get("chunk_overlap") or 80)
    if method == "tokens":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n","\n"," ",""])

def _embedder(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers for embedding.")
    return SentenceTransformer(model_name)

def _weaviate_client(cfg: Dict[str, Any]):
    if weaviate is None:
        raise RuntimeError("Install 'weaviate-client>=4,<5'")
    cluster_url = (cfg.get("cluster_url") or cfg.get("url") or "").strip().rstrip("/")
    api_key = (cfg.get("api_key") or "").strip() or None
    auth = Auth.api_key(api_key) if api_key else None
    if cluster_url:
        return connect_to_wcs(cluster_url=cluster_url, auth_credentials=auth)
    scheme = (cfg.get("scheme") or "https").lower()
    host = (cfg.get("host") or "").strip()
    port = int(cfg.get("port") or (443 if scheme == "https" else 80))
    if not host:
        raise RuntimeError("Weaviate host is required when cluster_url not provided.")
    return connect_to_custom(http_host=host, http_port=port, http_secure=(scheme=="https"), auth_credentials=auth)

def _ensure_collection(client, name: str, dims: int = 384, mt: bool=False):
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "KB"
    cols = {c.name: c for c in client.collections.list_all()}
    if name in cols: return client.collections.get(name)
    return client.collections.create(
        name,
        vectorizer_config=Configure.Vectorizer.none(dims=dims),
        properties=[
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
        ],
        multi_tenancy_config=Configure.MultiTenancy(enabled=bool(mt))
    )

# ---------- Main: run pipeline ----------
def run_pipeline_by_id(pipeline_id: str) -> Dict[str, Any]:
    pipelines = load_pipelines()
    p = pipelines.get(pipeline_id)
    if not p:
        raise ValueError(f"Pipeline {pipeline_id!r} not found.")

    conns = load_connections()

    # Resolve source & sink profiles
    src_id, src_profile = p["source"]["connector_id"], p["source"]["profile_name"]
    sink_id, sink_profile = p["sink"]["connector_id"], p["sink"]["profile_name"]

    src_cfg = (conns.get(src_id) or {}).get(src_profile) or {}
    sink_cfg = (conns.get(sink_id) or {}).get(sink_profile) or {}

    # 1) Fetch
    items: Iterable[Tuple[str, bytes]]
    fetch_cfg = p.get("source", {})
    if src_id == "azureblob":
        items = _iter_azblob_files(src_cfg, fetch_cfg.get("container",""), fetch_cfg.get("prefix",""), fetch_cfg.get("pattern", r".*"))
    elif src_id == "s3":
        raise NotImplementedError("S3 reader not included in this minimal runner.")
    elif src_id == "gcs":
        raise NotImplementedError("GCS reader not included in this minimal runner.")
    elif src_id == "localfs":
        items = _iter_local_files(fetch_cfg.get("root","./KB"), fetch_cfg.get("pattern", r".*"))
    else:
        raise NotImplementedError(f"Source {src_id} not supported by runner yet.")

    # 2) Chunk
    chunk_cfg = p.get("chunking", {"method":"recursive","chunk_size":800,"chunk_overlap":80})
    splitter = _make_splitter(chunk_cfg)

    docs: List[Dict[str, Any]] = []
    for path, data in items:
        txt = _bytes_to_text(path, data)
        if not txt: continue
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks):
            docs.append({"id": _sha1(f"{path}:{i}:{len(ch)}"), "path": path, "source": p["source"].get("name","src"), "chunk": ch})

    # 3) Embed
    emb_cfg = p.get("embedding", {"model":"sentence-transformers/all-MiniLM-L6-v2"})
    model = _embedder(emb_cfg.get("model","sentence-transformers/all-MiniLM-L6-v2"))
    vectors = model.encode([d["chunk"] for d in docs], batch_size=int(emb_cfg.get("batch_size", 64)), show_progress_bar=True)

    # 4) Upsert → Weaviate
    if sink_id != "weaviate":
        raise NotImplementedError("Only Weaviate sink implemented in this runner.")
    client = _weaviate_client(sink_cfg)
    try:
        coll_name = p["sink"].get("collection", "KB")
        dims = int(getattr(model, "get_sentence_embedding_dimension", lambda: len(vectors[0]))())
        mt = str(sink_cfg.get("multi_tenancy","")).lower() in ("true","1","yes")
        coll = _ensure_collection(client, coll_name, dims=dims, mt=mt)
        with coll.batch.dynamic() as batch:
            for vec, d in zip(vectors, docs):
                batch.add_object(properties={"source": d["source"], "chunk": d["chunk"], "path": d["path"]}, uuid=d["id"], vector=vec)
    finally:
        try: client.close()
        except Exception: pass

    return {"ingested": len(docs), "collection": coll_name, "source": src_id, "sink": sink_id}
