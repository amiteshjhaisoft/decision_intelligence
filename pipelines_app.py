"""
Author: Amitesh Jha | iSoft ANZ
Date: 2025-10-21
File: pipelines_app.py

One-file Streamlit app that behaves like a lightweight ETL tool for RAG pipelines:
- Define pipelines (Source → Chunking → Embedding → Vector DB sink)
- Save / Edit / Delete pipelines to pipelines.json
- Run pipelines end‑to‑end with progress + logs
- Pluggable connectors (Local Folder, Azure Blob, File Upload, PostgreSQL*)
- Vector stores: FAISS (local folder), Weaviate (cloud/local)
- Multiple chunking strategies + embedding providers

*Note: DB connectors beyond simple demo require drivers (psycopg2, snowflake-connector-python, etc.).
The runner includes graceful fallbacks so the UI remains responsive even if optional deps are missing.

Secrets supported (optional via .streamlit/secrets.toml):
[azure]
connection_string = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
container = "knowledgebase"
prefix = "KB"

[weaviate]
url = "http://localhost:8080"  # or your cluster URL
api_key = "YOUR-KEY"            # optional for local without auth

[openai]
api_key = "sk-..."              # if you want to use OpenAI embeddings

To run:
  streamlit run pipelines_app.py
"""

from __future__ import annotations

import os, io, json, time, uuid, glob, math, base64
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

# ---------------------- Optional heavy deps (all graceful) ----------------------
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    import docx2txt  # type: ignore
except Exception:
    docx2txt = None

try:
    import weaviate  # type: ignore
    import weaviate.classes as wvc  # type: ignore
except Exception:
    weaviate, wvc = None, None

# LangChain splitters (optional; we also ship a fallback splitter)
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        SentenceTransformersTokenTextSplitter,
        MarkdownTextSplitter,
    )  # type: ignore
except Exception:
    RecursiveCharacterTextSplitter = SentenceTransformersTokenTextSplitter = MarkdownTextSplitter = None

# FAISS for local vector store
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Sentence-Transformers (local embeddings)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

# OpenAI embeddings (optional)
try:
    import openai  # type: ignore
except Exception:
    openai = None

import streamlit as st

APP_TITLE = "RAG Pipelines — ETL‑style Builder"
PIPELINES_JSON = Path("pipelines.json")
LOCAL_INDEX_DIR = Path("./vector_indexes")
LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================================
# Data Models
# =====================================================================================
@dataclass
class SourceConfig:
    kind: str  # 'local_folder' | 'azure_blob' | 'file_upload' | 'postgres'
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkConfig:
    strategy: str = "recursive"  # 'recursive' | 'token' | 'markdown' | 'fixed'
    chunk_size: int = 800
    chunk_overlap: int = 120
    token_chunk_size: int = 256
    token_overlap: int = 32

@dataclass
class EmbeddingConfig:
    provider: str = "sentence_transformers"  # 'sentence_transformers' | 'openai'
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True

@dataclass
class VectorDBConfig:
    kind: str = "faiss"  # 'faiss' | 'weaviate'
    params: Dict[str, Any] = field(default_factory=dict)  # faiss: {index_path}; weaviate: {url, api_key, class_name}

@dataclass
class Pipeline:
    id: str
    name: str
    description: str
    source: SourceConfig
    chunking: ChunkConfig
    embedding: EmbeddingConfig
    vectordb: VectorDBConfig

# =====================================================================================
# Persistence Helpers
# =====================================================================================

def load_pipelines() -> Dict[str, Pipeline]:
    if not PIPELINES_JSON.exists():
        return {}
    try:
        raw = json.loads(PIPELINES_JSON.read_text(encoding="utf-8"))
        out: Dict[str, Pipeline] = {}
        for pid, pdata in raw.items():
            out[pid] = Pipeline(
                id=pdata["id"],
                name=pdata.get("name", pid),
                description=pdata.get("description", ""),
                source=SourceConfig(**pdata["source"]),
                chunking=ChunkConfig(**pdata["chunking"]),
                embedding=EmbeddingConfig(**pdata["embedding"]),
                vectordb=VectorDBConfig(**pdata["vectordb"]),
            )
        return out
    except Exception as e:
        st.warning(f"Could not read pipelines.json: {e}")
        return {}


def save_pipelines(store: Dict[str, Pipeline]) -> None:
    serial = {pid: asdict(p) for pid, p in store.items()}
    PIPELINES_JSON.write_text(json.dumps(serial, indent=2), encoding="utf-8")

# =====================================================================================
# UI Helpers
# =====================================================================================

def _pill(text: str) -> str:
    return f"<span style='background:#EEF2FF;color:#3730A3;padding:2px 8px;border-radius:9999px;font-size:12px'>{text}</span>"


def sidebar_info():
    with st.sidebar:
        st.subheader("Connections")
        st.caption("Vector DB connections are used at sync step. Configure via form when selecting a sink.")
        st.markdown("- **FAISS**: stored under `./vector_indexes/`\n- **Weaviate**: uses secrets or form fields")
        st.divider()
        st.subheader("Docs")
        st.markdown("Use the form to define your pipeline. Save, then **Run**. Pipelines are JSON‑backed.")

# =====================================================================================
# Source Readers
# =====================================================================================

def read_source_texts(src: SourceConfig, log) -> List[Tuple[str, str]]:
    """Return list of (doc_id, text)."""
    kind = src.kind
    p = src.params or {}
    texts: List[Tuple[str, str]] = []

    def _add(path: str, text: str):
        if text and text.strip():
            texts.append((path, text))

    if kind == "local_folder":
        root = p.get("path") or "."
        exts = set((p.get("exts") or ".txt,.md,.pdf,.docx,.csv,.json").split(","))
        paths = []
        for ext in exts:
            paths.extend(glob.glob(str(Path(root) / f"**/*{ext.strip()}"), recursive=True))
        for fp in paths:
            try:
                ext = Path(fp).suffix.lower()
                if ext in {".txt", ".md"}:
                    _add(fp, Path(fp).read_text(encoding="utf-8", errors="ignore"))
                elif ext == ".pdf" and PdfReader:
                    reader = PdfReader(fp)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    _add(fp, text)
                elif ext == ".docx" and docx2txt:
                    _add(fp, docx2txt.process(fp) or "")
                elif ext == ".csv":
                    if pd is not None:
                        df = pd.read_csv(fp)
                        _add(fp, df.to_csv(index=False))
                elif ext == ".json":
                    _add(fp, Path(fp).read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                log(f"⚠️ Skipped {fp}: {e}")
        log(f"Loaded {len(texts)} files from local folder")

    elif kind == "azure_blob":
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except Exception:
            raise RuntimeError("azure-storage-blob not installed. pip install azure-storage-blob")
        conn = p.get("connection_string") or st.secrets.get("azure", {}).get("connection_string")
        container = p.get("container") or st.secrets.get("azure", {}).get("container")
        prefix = p.get("prefix") or st.secrets.get("azure", {}).get("prefix", "")
        if not conn or not container:
            raise RuntimeError("Azure connection_string and container are required.")
        bsc = BlobServiceClient.from_connection_string(conn)
        cont = bsc.get_container_client(container)
        blobs = cont.list_blobs(name_starts_with=prefix)
        for b in blobs:
            name = b.name
            if name.endswith("/"):
                continue
            try:
                data = cont.download_blob(name).readall()
                ext = Path(name).suffix.lower()
                if ext in {".txt", ".md", ".json", ".csv"}:
                    text = data.decode("utf-8", errors="ignore")
                    if ext == ".csv" and pd is not None:
                        try:
                            from io import StringIO
                            df = pd.read_csv(StringIO(text))
                            text = df.to_csv(index=False)
                        except Exception:
                            pass
                    _add(name, text)
                elif ext == ".pdf" and PdfReader:
                    with io.BytesIO(data) as bio:
                        reader = PdfReader(bio)
                        t = "\n".join([pg.extract_text() or "" for pg in reader.pages])
                        _add(name, t)
                elif ext == ".docx" and docx2txt:
                    with io.BytesIO(data) as bio:
                        tmp = f"/tmp/{uuid.uuid4().hex}.docx"
                        Path(tmp).write_bytes(bio.read())
                        _add(name, docx2txt.process(tmp) or "")
                        try: Path(tmp).unlink(missing_ok=True)
                        except Exception: pass
            except Exception as e:
                log(f"⚠️ Skip blob {name}: {e}")
        log(f"Loaded {len(texts)} blobs from Azure")

    elif kind == "file_upload":
        # st.file_uploader handles UI; here we just accept bytes in params
        files: List[Tuple[str, bytes]] = p.get("files", [])
        for fname, data in files:
            try:
                ext = Path(fname).suffix.lower()
                if ext in {".txt", ".md", ".json", ".csv"}:
                    text = data.decode("utf-8", errors="ignore")
                    if ext == ".csv" and pd is not None:
                        try:
                            from io import StringIO
                            df = pd.read_csv(StringIO(text))
                            text = df.to_csv(index=False)
                        except Exception:
                            pass
                    _add(fname, text)
                elif ext == ".pdf" and PdfReader:
                    with io.BytesIO(data) as bio:
                        reader = PdfReader(bio)
                        t = "\n".join([pg.extract_text() or "" for pg in reader.pages])
                        _add(fname, t)
                elif ext == ".docx" and docx2txt:
                    tmp = f"/tmp/{uuid.uuid4().hex}.docx"
                    Path(tmp).write_bytes(data)
                    _add(fname, docx2txt.process(tmp) or "")
                    try: Path(tmp).unlink(missing_ok=True)
                    except Exception: pass
            except Exception as e:
                log(f"⚠️ Skip upload {fname}: {e}")
        log(f"Loaded {len(texts)} uploaded files")

    elif kind == "postgres":
        try:
            import psycopg2  # type: ignore
        except Exception:
            raise RuntimeError("psycopg2 not installed. pip install psycopg2-binary")
        host = p.get("host"); db = p.get("database"); user = p.get("user"); pw = p.get("password"); q = p.get("query")
        if not all([host, db, user, pw, q]):
            raise RuntimeError("PostgreSQL host, database, user, password, and SQL query are required.")
        conn = psycopg2.connect(host=host, dbname=db, user=user, password=pw)
        cur = conn.cursor()
        cur.execute(q)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if pd is not None:
            df = pd.DataFrame(rows, columns=cols)
            _add("postgres_query", df.to_csv(index=False))
        else:
            text_rows = ["\t".join(map(str, r)) for r in rows]
            _add("postgres_query", "\n".join(text_rows))
        cur.close(); conn.close()
        log(f"Loaded {len(rows)} rows from PostgreSQL")

    else:
        raise RuntimeError(f"Unsupported source kind: {kind}")

    return texts

# =====================================================================================
# Chunking
# =====================================================================================

def chunk_texts(texts: List[Tuple[str, str]], cfg: ChunkConfig, log) -> List[Tuple[str, str]]:
    """Return list of (chunk_id, chunk_text)."""
    chunks: List[Tuple[str, str]] = []

    def fallback_split(text: str, size: int, overlap: int) -> List[str]:
        out = []
        i = 0
        n = len(text)
        while i < n:
            out.append(text[i : i + size])
            i += max(1, size - overlap)
        return out

    for doc_id, text in texts:
        if not text.strip():
            continue
        if cfg.strategy == "recursive" and RecursiveCharacterTextSplitter:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                separators=["\n## ", "\n# ", "\n", " ", ""]
            )
            parts = splitter.split_text(text)
        elif cfg.strategy == "markdown" and MarkdownTextSplitter:
            splitter = MarkdownTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
            parts = splitter.split_text(text)
        elif cfg.strategy == "token" and SentenceTransformersTokenTextSplitter:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_size=cfg.token_chunk_size,
                chunk_overlap=cfg.token_overlap,
            )
            parts = splitter.split_text(text)
        elif cfg.strategy == "fixed":
            parts = fallback_split(text, cfg.chunk_size, cfg.chunk_overlap)
        else:
            parts = fallback_split(text, cfg.chunk_size, cfg.chunk_overlap)

        for j, p in enumerate(parts):
            chunks.append((f"{doc_id}::chunk{j}", p))

    log(f"Created {len(chunks)} chunks using '{cfg.strategy}' strategy")
    return chunks

# =====================================================================================
# Embeddings
# =====================================================================================

def embed_chunks(chunks: List[Tuple[str, str]], cfg: EmbeddingConfig, log) -> Tuple[List[str], List[List[float]]]:
    ids = [cid for cid, _ in chunks]
    texts = [t for _, t in chunks]

    if cfg.provider == "sentence_transformers":
        if not SentenceTransformer:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        model = SentenceTransformer(cfg.model_name)
        vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=cfg.normalize)
        log(f"Embedded {len(texts)} chunks with Sentence-Transformers: {cfg.model_name}")
        return ids, vecs.tolist()

    elif cfg.provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in env or secrets")
        openai.api_key = api_key
        # Use text-embedding-3-small by default
        model_name = cfg.model_name or "text-embedding-3-small"
        # Batch for reliability
        vecs: List[List[float]] = []
        B = 1000
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            resp = openai.embeddings.create(model=model_name, input=batch)
            vecs.extend([d.embedding for d in resp.data])
        log(f"Embedded {len(texts)} chunks with OpenAI: {model_name}")
        return ids, vecs

    else:
        raise RuntimeError(f"Unsupported embedding provider: {cfg.provider}")

# =====================================================================================
# Vector DB Upsert
# =====================================================================================

def upsert_to_vdb(ids: List[str], vecs: List[List[float]], chunks: List[Tuple[str, str]], sink: VectorDBConfig, log):
    if sink.kind == "faiss":
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed. pip install faiss-cpu")
        import numpy as np  # local import to avoid mandatory dep at import time
        arr = np.array(vecs, dtype="float32")
        d = arr.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine-like if vectors normalized
        index.add(arr)
        out_dir = Path(sink.params.get("index_path") or (LOCAL_INDEX_DIR / f"{uuid.uuid4().hex}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        # Also persist mapping id→text for simple retrieval demos
        meta = {i: {"id": ids[k], "text": chunks[k][1]} for k, i in enumerate(range(len(ids)))}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Upserted {len(ids)} vectors to FAISS at {out_dir}")
        return {"index_path": str(out_dir)}

    elif sink.kind == "weaviate":
        if weaviate is None:
            raise RuntimeError("weaviate-client not installed. pip install weaviate-client")
        url = sink.params.get("url") or st.secrets.get("weaviate", {}).get("url")
        api_key = sink.params.get("api_key") or st.secrets.get("weaviate", {}).get("api_key")
        clazz = sink.params.get("class_name") or "KBChunk"
        dim = len(vecs[0]) if vecs else 384
        if not url:
            raise RuntimeError("Weaviate URL is required")
        if api_key:
            client = weaviate.WeaviateClient(
                connection_params=weaviate.ConnectionParams.from_url(url),
                auth_client_secret=weaviate.auth.AuthApiKey(api_key=api_key),
                skip_init_checks=True,
            )
        else:
            client = weaviate.WeaviateClient(
                connection_params=weaviate.ConnectionParams.from_url(url),
                skip_init_checks=True,
            )
        # schema ensure
        existing = [c.class_name for c in client.collections.list_all()] if hasattr(client, "collections") else []
        if clazz not in existing:
            client.collections.create(
                name=clazz,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
                properties=[
                    wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                ],
                vector_dimensions=dim,
            )
        coll = client.collections.get(clazz)
        # batch upsert
        with coll.batch.dynamic() as batch:
            for i, (cid, txt) in enumerate(chunks):
                batch.add_object(
                    properties={"doc_id": cid, "text": txt},
                    vector=vecs[i],
                    uuid=uuid.uuid5(uuid.NAMESPACE_URL, cid).hex,
                )
        log(f"Upserted {len(ids)} vectors to Weaviate class '{clazz}' @ {url}")
        return {"class_name": clazz, "url": url}

    else:
        raise RuntimeError(f"Unsupported vector DB kind: {sink.kind}")

# =====================================================================================
# Runner
# =====================================================================================

def run_pipeline(p: Pipeline, uploads: Optional[List[Any]] = None, ui_log=None) -> Dict[str, Any]:
    def log(msg: str):
        if ui_log:
            ui_log(msg)
        else:
            print(msg)

    log(f"▶️ Running pipeline: {p.name}")

    src = p.source
    if src.kind == "file_upload" and uploads:
        # insert actual uploaded files into params for this run
        src = SourceConfig(kind="file_upload", params={"files": [(u.name, u.getvalue()) for u in uploads]})

    with st.status("Fetching source data…", expanded=True) as s1:
        texts = read_source_texts(src, log)
        s1.update(label=f"Fetched {len(texts)} docs", state="complete")

    with st.status("Chunking…", expanded=False) as s2:
        chunks = chunk_texts(texts, p.chunking, log)
        s2.update(label=f"Created {len(chunks)} chunks", state="complete")

    with st.status("Embedding…", expanded=False) as s3:
        ids, vecs = embed_chunks(chunks, p.embedding, log)
        s3.update(label=f"Embedded {len(ids)} chunks", state="complete")

    with st.status("Upserting to Vector DB…", expanded=False) as s4:
        out = upsert_to_vdb(ids, vecs, chunks, p.vectordb, log)
        s4.update(label="Completed upsert", state="complete")

    log("✅ Pipeline completed")
    return {"documents": len(texts), "chunks": len(ids), "sink_info": out}

# =====================================================================================
# UI — App
# =====================================================================================

def _init_state():
    if "pipelines" not in st.session_state:
        st.session_state.pipelines = load_pipelines()
    if "editing_id" not in st.session_state:
        st.session_state.editing_id = None


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")
    _init_state()
    sidebar_info()

    st.title(APP_TITLE)
    st.caption("Define and run end‑to‑end pipelines for your RAG Knowledge Base.")

    left, right = st.columns([0.55, 0.45], gap="large")

    # ---------------------------- Existing Pipelines (List) ----------------------------
    with left:
        st.subheader("Pipelines")
        store: Dict[str, Pipeline] = st.session_state.pipelines
        if not store:
            st.info("No pipelines yet. Create one on the right.")
        else:
            for pid, p in list(store.items()):
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([0.55, 0.15, 0.15, 0.15])
                    with c1:
                        st.markdown(f"**{p.name}**  ")
                        st.caption(p.description or "(no description)")
                        st.markdown(
                            f"{_pill(p.source.kind)} {_pill(p.chunking.strategy)} {_pill(p.embedding.provider)} {_pill(p.vectordb.kind)}",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button("Run", key=f"run_{pid}"):
                            try:
                                res = run_pipeline(p, ui_log=lambda m: st.write(m))
                                st.success(f"Completed: {res}")
                            except Exception as e:
                                st.error(f"Run failed: {e}")
                    with c3:
                        if st.button("Edit", key=f"edit_{pid}"):
                            st.session_state.editing_id = pid
                    with c4:
                        if st.button("Delete", key=f"del_{pid}"):
                            store.pop(pid, None)
                            save_pipelines(store)
                            st.toast("Pipeline deleted", icon="🗑️")
                            st.rerun()

    # ---------------------------- Create / Edit Form ----------------------------
    with right:
        st.subheader("Create / Edit Pipeline")

        editing_id = st.session_state.editing_id
        editing: Optional[Pipeline] = None
        if editing_id and editing_id in st.session_state.pipelines:
            editing = st.session_state.pipelines[editing_id]

        with st.form("pipeline_form", clear_on_submit=False):
            name = st.text_input("Name", value=(editing.name if editing else "My Pipeline"))
            desc = st.text_area("Description", value=(editing.description if editing else ""), height=80)

            st.markdown("### Source")
            src_kind = st.selectbox(
                "Source type",
                ["local_folder", "azure_blob", "file_upload", "postgres"],
                index=(
                    ["local_folder", "azure_blob", "file_upload", "postgres"].index(editing.source.kind)
                    if editing else 0
                ),
                help="Pick where to fetch data from.",
            )
            src_params: Dict[str, Any] = {}
            if src_kind == "local_folder":
                src_params["path"] = st.text_input("Folder path", value=(editing.source.params.get("path") if editing and editing.source.kind=="local_folder" else "./KB"))
                src_params["exts"] = st.text_input("Extensions (comma‑sep)", value=(editing.source.params.get("exts") if editing and editing.source.kind=="local_folder" else ".txt,.md,.pdf,.docx,.csv,.json"))
            elif src_kind == "azure_blob":
                az = st.secrets.get("azure", {})
                src_params["connection_string"] = st.text_input("Azure connection_string", value=(editing.source.params.get("connection_string") if editing and editing.source.kind=="azure_blob" else az.get("connection_string", "")))
                src_params["container"] = st.text_input("Container", value=(editing.source.params.get("container") if editing and editing.source.kind=="azure_blob" else az.get("container", "")))
                src_params["prefix"] = st.text_input("Prefix (folder)", value=(editing.source.params.get("prefix") if editing and editing.source.kind=="azure_blob" else az.get("prefix", "")))
            elif src_kind == "file_upload":
                st.caption("Upload files now; they are stored only for this run, not in pipelines.json.")
                up = st.file_uploader("Upload documents", type=["txt","md","pdf","docx","csv","json"], accept_multiple_files=True)
                # we pass these during run, not saved
                src_params["note"] = "Files are provided at run time"
            elif src_kind == "postgres":
                col1, col2 = st.columns(2)
                with col1:
                    host = st.text_input("Host", value=(editing.source.params.get("host") if editing and editing.source.kind=="postgres" else ""))
                    db = st.text_input("Database", value=(editing.source.params.get("database") if editing and editing.source.kind=="postgres" else ""))
                    user = st.text_input("User", value=(editing.source.params.get("user") if editing and editing.source.kind=="postgres" else ""))
                with col2:
                    pw = st.text_input("Password", type="password", value=(editing.source.params.get("password") if editing and editing.source.kind=="postgres" else ""))
                    query = st.text_area("SQL Query", value=(editing.source.params.get("query") if editing and editing.source.kind=="postgres" else "SELECT * FROM public.mytable LIMIT 100;"))
                src_params.update({"host": host, "database": db, "user": user, "password": pw, "query": query})

            st.markdown("### Chunking")
            strategy = st.selectbox("Strategy", ["recursive","markdown","token","fixed"], index=(
                ["recursive","markdown","token","fixed"].index(editing.chunking.strategy) if editing else 0
            ))
            c1, c2 = st.columns(2)
            with c1:
                chunk_size = st.number_input("Chunk size (chars)", 100, 5000, value=(editing.chunking.chunk_size if editing else 800), step=50)
            with c2:
                chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 1000, value=(editing.chunking.chunk_overlap if editing else 120), step=10)
            c3, c4 = st.columns(2)
            with c3:
                token_chunk_size = st.number_input("Token chunk size", 32, 2048, value=(editing.chunking.token_chunk_size if editing else 256), step=16)
            with c4:
                token_overlap = st.number_input("Token overlap", 0, 512, value=(editing.chunking.token_overlap if editing else 32), step=8)

            st.markdown("### Embeddings")
            provider = st.selectbox("Provider", ["sentence_transformers","openai"], index=(
                ["sentence_transformers","openai"].index(editing.embedding.provider) if editing else 0
            ))
            model_name = st.text_input("Model name", value=(editing.embedding.model_name if editing else "sentence-transformers/all-MiniLM-L6-v2"))
            normalize = st.checkbox("Normalize embeddings (cosine)", value=(editing.embedding.normalize if editing else True))

            st.markdown("### Vector DB (Sink)")
            sink_kind = st.selectbox("Vector DB", ["faiss","weaviate"], index=(
                ["faiss","weaviate"].index(editing.vectordb.kind) if editing else 0
            ), help="Only Vector DB connections are used at sync stage.")
            sink_params: Dict[str, Any] = {}
            if sink_kind == "faiss":
                sink_params["index_path"] = st.text_input("Index output dir", value=(editing.vectordb.params.get("index_path") if editing and editing.vectordb.kind=="faiss" else str(LOCAL_INDEX_DIR / "kb_index")))
            elif sink_kind == "weaviate":
                w = st.secrets.get("weaviate", {})
                sink_params["url"] = st.text_input("Weaviate URL", value=(editing.vectordb.params.get("url") if editing and editing.vectordb.kind=="weaviate" else w.get("url", "http://localhost:8080")))
                sink_params["api_key"] = st.text_input("API Key (optional)", value=(editing.vectordb.params.get("api_key") if editing and editing.vectordb.kind=="weaviate" else w.get("api_key", "")))
                sink_params["class_name"] = st.text_input("Class name", value=(editing.vectordb.params.get("class_name") if editing and editing.vectordb.kind=="weaviate" else "KBChunk"))

            submitted = st.form_submit_button("Save Pipeline")

        if submitted:
            pid = editing.id if editing else uuid.uuid4().hex
            pipe = Pipeline(
                id=pid,
                name=name.strip(),
                description=desc.strip(),
                source=SourceConfig(kind=src_kind, params=src_params),
                chunking=ChunkConfig(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap, token_chunk_size=token_chunk_size, token_overlap=token_overlap),
                embedding=EmbeddingConfig(provider=provider, model_name=model_name.strip(), normalize=normalize),
                vectordb=VectorDBConfig(kind=sink_kind, params=sink_params),
            )
            st.session_state.pipelines[pid] = pipe
            save_pipelines(st.session_state.pipelines)
            st.session_state.editing_id = None
            st.success("Pipeline saved.")
            st.rerun()

        # Quick Run (for upload sources)
        with st.expander("Quick Run (without saving)"):
            q_src = st.selectbox("Source", ["file_upload","local_folder","azure_blob"], index=0)
            q_params: Dict[str, Any] = {}
            uploads = None
            if q_src == "file_upload":
                uploads = st.file_uploader("Upload docs", type=["txt","md","pdf","docx","csv","json"], accept_multiple_files=True, key="quick_upload")
            elif q_src == "local_folder":
                q_params["path"] = st.text_input("Folder path", "./KB", key="quick_local")
            elif q_src == "azure_blob":
                az = st.secrets.get("azure", {})
                q_params["connection_string"] = st.text_input("Azure connection_string", az.get("connection_string", ""), key="quick_conn")
                q_params["container"] = st.text_input("Container", az.get("container", ""), key="quick_cont")
                q_params["prefix"] = st.text_input("Prefix", az.get("prefix", ""), key="quick_pref")

            q_embed = st.selectbox("Embeddings", ["sentence-transformers/all-MiniLM-L6-v2","openai:text-embedding-3-small"], index=0)
            if q_embed.startswith("openai"):
                eprov = "openai"; emodel = q_embed.split(":",1)[1]
            else:
                eprov = "sentence_transformers"; emodel = q_embed
            q_sink = st.selectbox("Sink", ["faiss","weaviate"], index=0)
            sink_p = {}
            if q_sink == "faiss":
                sink_p["index_path"] = st.text_input("Index output dir", str(LOCAL_INDEX_DIR / "quick_index"), key="quick_faiss")
            else:
                w = st.secrets.get("weaviate", {})
                sink_p["url"] = st.text_input("Weaviate URL", w.get("url", "http://localhost:8080"), key="quick_w_url")
                sink_p["api_key"] = st.text_input("API Key (optional)", w.get("api_key", ""), key="quick_w_key")
                sink_p["class_name"] = st.text_input("Class name", "KBChunk", key="quick_w_class")

            if st.button("Run Quick Pipeline"):
                qp = Pipeline(
                    id=uuid.uuid4().hex,
                    name="Quick Run",
                    description="",
                    source=SourceConfig(kind=q_src, params=q_params),
                    chunking=ChunkConfig(),
                    embedding=EmbeddingConfig(provider=eprov, model_name=emodel),
                    vectordb=VectorDBConfig(kind=q_sink, params=sink_p),
                )
                try:
                    res = run_pipeline(qp, uploads=uploads, ui_log=lambda m: st.write(m))
                    st.success(f"Quick run done: {res}")
                except Exception as e:
                    st.error(f"Quick run failed: {e}")


if __name__ == "__main__":
    main()
