# ======================= pipelines_inline.py (can be pasted inline) =======================
# ADF-style ingest → chunk → embed → sync-to-VectorDB (Weaviate)
# - Drop into connectors_hub.py (inline) or keep as separate module and import.
# - Uses existing connections.json and a new pipelines.json in the same folder.

from __future__ import annotations
import os, io, re, json, time, hashlib, threading
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

# ------------------------------ CONFIG / LOCATIONS ------------------------------
APP_DIR = Path(__file__).parent
CONN_STORE = APP_DIR / "connections.json"
PIPE_STORE = APP_DIR / "pipelines.json"

# ------------------------------ STORES (1) --------------------------------------
def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(p: Path, d: Dict[str, Any]) -> None:
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")

def load_connections() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return _load_json(CONN_STORE)

def save_connections(d: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    _save_json(CONN_STORE, d)

def load_pipelines() -> Dict[str, Any]:
    return _load_json(PIPE_STORE)

def save_pipelines(d: Dict[str, Any]) -> None:
    _save_json(PIPE_STORE, d)

# ------------------------------ RUNNER (2) --------------------------------------
# Optional deps kept graceful:
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

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _iter_local_files(root: str, pattern: str = r".*\.(txt|md|csv|json|pdf|docx|xlsx)$") -> Iterable[Tuple[str, bytes]]:
    rootp = Path(root)
    rx = re.compile(pattern, re.I)
    for p in rootp.rglob("*"):
        if p.is_file() and rx.match(p.name):
            try:
                yield (str(p), p.read_bytes())
            except Exception:
                pass

def _iter_azblob_files(cfg: Dict[str, Any], container: str, prefix: str = "", pattern: str = r".*") -> Iterable[Tuple[str, bytes]]:
    if BlobServiceClient is None:
        raise RuntimeError("Install azure-storage-blob for Azure Blob source.")
    # Accept any of: connection_string | sas_url | (account_name+account_key)
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
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

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
    return connect_to_custom(
        http_host=host, http_port=port, http_secure=(scheme == "https"),
        auth_credentials=auth
    )

def _ensure_collection(client, name: str, dims: int = 384, mt: bool=False):
    # sanitize collection name for Weaviate
    name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "KB"
    cols = {c.name: c for c in client.collections.list_all()}
    if name in cols:
        return client.collections.get(name)
    return client.collections.create(
        name,
        vectorizer_config=Configure.Vectorizer.none(dims=dims),
        properties=[
            Property(name="source", data_type=DataType.TEXT),
            Property(name="chunk",  data_type=DataType.TEXT),
            Property(name="path",   data_type=DataType.TEXT),
        ],
        multi_tenancy_config=Configure.MultiTenancy(enabled=bool(mt))
    )

def run_pipeline_dict(p: Dict[str, Any]) -> Dict[str, Any]:
    """Run a pipeline object (already materialized as dict)."""
    conns = load_connections()

    # Resolve source & sink profiles
    src_id, src_profile = p["source"]["connector_id"], p["source"]["profile_name"]
    sink_id, sink_profile = p["sink"]["connector_id"], p["sink"]["profile_name"]

    src_cfg = (conns.get(src_id) or {}).get(src_profile) or {}
    sink_cfg = (conns.get(sink_id) or {}).get(sink_profile) or {}

    # 1) Fetch
    fetch_cfg = p.get("source", {})
    if src_id == "azureblob":
        items: Iterable[Tuple[str, bytes]] = _iter_azblob_files(
            src_cfg,
            fetch_cfg.get("container",""),
            fetch_cfg.get("prefix",""),
            fetch_cfg.get("pattern", r".*"),
        )
    elif src_id == "localfs":
        items = _iter_local_files(
            fetch_cfg.get("root","./KB"),
            fetch_cfg.get("pattern", r".*"),
        )
    else:
        raise NotImplementedError(f"Source {src_id} not supported.")

    # 2) Chunk
    chunk_cfg = p.get("chunking", {"method":"recursive","chunk_size":800,"chunk_overlap":80})
    splitter = _make_splitter(chunk_cfg)

    docs: List[Dict[str, Any]] = []
    for path, data in items:
        txt = _bytes_to_text(path, data)
        if not txt: 
            continue
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": _sha1(f"{path}:{i}:{len(ch)}"),
                "path": path,
                "source": p["source"].get("name","src"),
                "chunk": ch
            })

    # 3) Embed
    emb_cfg = p.get("embedding", {"model":"sentence-transformers/all-MiniLM-L6-v2"})
    model_name = emb_cfg.get("model","sentence-transformers/all-MiniLM-L6-v2")
    model = _embedder(model_name)
    vectors = model.encode(
        [d["chunk"] for d in docs],
        batch_size=int(emb_cfg.get("batch_size", 64)),
        show_progress_bar=True
    )

    # 4) Upsert to Weaviate
    if sink_id != "weaviate":
        raise NotImplementedError("Only Weaviate sink implemented.")
    client = _weaviate_client(sink_cfg)
    try:
        coll_name = p["sink"].get("collection", "KB")
        dims = int(getattr(model, "get_sentence_embedding_dimension", lambda: len(vectors[0]))())
        mt = str(sink_cfg.get("multi_tenancy","")).lower() in ("true","1","yes")
        coll = _ensure_collection(client, coll_name, dims=dims, mt=mt)
        cnt = 0
        with coll.batch.dynamic() as batch:
            for vec, d in zip(vectors, docs):
                batch.add_object(
                    properties={"source": d["source"], "chunk": d["chunk"], "path": d["path"]},
                    uuid=d["id"], vector=vec
                )
                cnt += 1
    finally:
        try: client.close()
        except Exception: pass

    return {"ingested": len(docs), "collection": coll_name, "source": src_id, "sink": sink_id}

def run_pipeline_by_id(pipeline_id: str) -> Dict[str, Any]:
    pipelines = load_pipelines()
    p = pipelines.get(pipeline_id)
    if not p:
        raise ValueError(f"Pipeline {pipeline_id!r} not found.")
    return run_pipeline_dict(p)

# ------------------------------ TRIGGER LOOP (4) --------------------------------
def trigger_loop_once() -> bool:
    """
    Run due interval pipelines once (non-blocking helper for external schedulers).
    Returns True if any pipeline state changed.
    """
    from datetime import datetime, timedelta
    pipes = load_pipelines()
    changed = False
    now = datetime.utcnow()
    for pid, p in pipes.items():
        trig = p.get("trigger", {"type":"manual"})
        if trig.get("type") != "interval":
            continue
        every = int(trig.get("interval_minutes", 60))
        last = None
        if p.get("_last_run_utc"):
            try:
                last = datetime.fromisoformat(p["_last_run_utc"])
            except Exception:
                last = None
        due = (last is None) or (now - last >= timedelta(minutes=every))
        if due:
            try:
                run_pipeline_dict(p)
                p["_last_run_utc"] = now.isoformat()
                changed = True
            except Exception as e:
                # Best-effort logging
                print(f"[trigger] Run failed for {pid}: {e}")
    if changed:
        save_pipelines(pipes)
    return changed

def trigger_loop_forever(poll_seconds: int = 30):
    """Blocking loop you can launch in a background process or scheduler."""
    while True:
        try:
            trigger_loop_once()
        except Exception as e:
            print(f"[trigger] loop error: {e}")
        time.sleep(poll_seconds)

# ------------------------------ UI (3) ------------------------------------------
# This section is safe to call inline from Streamlit-enabled connectors_hub.py
def render_pipelines_ui():
    try:
        import streamlit as st
    except Exception:
        raise RuntimeError("Streamlit is required to render the pipelines UI.")

    st.markdown("## 🛠️ Pipelines")
    st.caption("Design pipelines that extract → chunk → embed → sync to your Vector DB (ADF-style).")

    conns = load_connections()
    pipes = load_pipelines()

    def _defaults() -> Dict[str, Any]:
        return {
            "name": "new_pipeline",
            "source": {
                "connector_id": "azureblob",   # or "localfs"
                "profile_name": "",
                "container": "", "prefix": "", "pattern": r".*\.(txt|md|pdf|docx|xlsx|csv|json)$",
                "root": "./KB",  # only for localfs
                "name": "KB"
            },
            "chunking": {"method":"recursive","chunk_size":800,"chunk_overlap":80},
            "embedding": {"model":"sentence-transformers/all-MiniLM-L6-v2","batch_size":64},
            "sink": {"connector_id":"weaviate","profile_name":"","collection":"KB"},
            "trigger": {"type":"manual","interval_minutes": 0}
        }

    def _list_profiles(conn_id: str) -> List[str]:
        return sorted(list((conns.get(conn_id) or {}).keys()))

    left, right = st.columns([4,6], gap="large")

    # -------- Left: Saved pipelines with Run/Delete --------
    with left:
        st.markdown("### 📜 Saved pipelines")
        if not pipes:
            st.info("No pipelines yet. Create one on the right.")
        else:
            for pid, meta in sorted(pipes.items(), key=lambda kv: kv[1].get("name","").lower()):
                c1, c2, c3 = st.columns([6,2,2])
                c1.write(f"**{meta.get('name','pipeline')}**  \n`{pid}`")
                if c2.button("▶️ Run", key=f"run::{pid}", help="Run pipeline now", use_container_width=True):
                    with st.spinner("Running..."):
                        try:
                            result = run_pipeline_by_id(pid)
                            st.success(f"Ingested {result['ingested']} chunks → collection **{result['collection']}**.")
                        except Exception as e:
                            st.error(f"Run failed: {e}")
                if c3.button("🗑️", key=f"del::{pid}", help="Delete pipeline", use_container_width=True):
                    try:
                        pipes.pop(pid, None)
                        save_pipelines(pipes)
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    # -------- Right: Editor (Create/Edit) --------
    with right:
        st.markdown("### ✏️ Create / Edit pipeline")
        mode = st.radio("Mode", ["Create new", "Edit existing"], horizontal=True)
        if mode == "Edit existing" and pipes:
            sel_id = st.selectbox("Select pipeline", list(pipes.keys()))
            draft = json.loads(json.dumps(pipes[sel_id]))  # deep copy
        elif mode == "Edit existing" and not pipes:
            st.warning("No pipelines to edit; switch to Create new.")
            draft = _defaults()
        else:
            draft = _defaults()

        draft["name"] = st.text_input("Pipeline name", draft.get("name","pipeline_1"))
        st.divider()

        st.subheader("Source")
        src_id = st.selectbox(
            "Source connector", ["azureblob","localfs"],
            index=0 if draft["source"]["connector_id"]=="azureblob" else 1,
            help="Pick from your saved connections"
        )
        draft["source"]["connector_id"] = src_id
        src_profiles = _list_profiles(src_id)
        draft["source"]["profile_name"] = st.selectbox(
            "Source profile", src_profiles,
            index=0 if src_profiles else None, placeholder="Select a saved profile"
        ) if src_profiles else ""

        if src_id == "azureblob":
            draft["source"]["container"] = st.text_input("Container", draft["source"].get("container",""))
            draft["source"]["prefix"] = st.text_input("Prefix (optional)", draft["source"].get("prefix",""))
            draft["source"]["pattern"] = st.text_input("Filename regex", draft["source"].get("pattern", r".*"))
        else:
            draft["source"]["root"] = st.text_input("Local folder path", draft["source"].get("root","./KB"))
            draft["source"]["pattern"] = st.text_input("Filename regex", draft["source"].get("pattern", r".*"))

        draft["source"]["name"] = st.text_input("Source label (stored in metadata)", draft["source"].get("name","KB"))

        st.subheader("Chunking")
        c1, c2, c3 = st.columns(3)
        draft["chunking"]["method"] = c1.selectbox(
            "Method", ["recursive","tokens"],
            index=0 if draft["chunking"].get("method","recursive")=="recursive" else 1
        )
        draft["chunking"]["chunk_size"] = int(c2.number_input("Chunk size", 100, 5000, value=int(draft["chunking"].get("chunk_size",800)), step=50))
        draft["chunking"]["chunk_overlap"] = int(c3.number_input("Overlap", 0, 1000, value=int(draft["chunking"].get("chunk_overlap",80)), step=10))

        st.subheader("Embedding")
        e1, e2 = st.columns([3,1])
        draft["embedding"]["model"] = e1.text_input("Sentence-Transformers model", draft["embedding"].get("model","sentence-transformers/all-MiniLM-L6-v2"))
        draft["embedding"]["batch_size"] = int(e2.number_input("Batch size", 1, 4096, value=int(draft["embedding"].get("batch_size",64)), step=1))

        st.subheader("Sink (Vector DB)")
        sink_id = st.selectbox("Sink connector", ["weaviate"], index=0, help="Choose a saved Weaviate profile")
        draft["sink"]["connector_id"] = sink_id
        sink_profiles = _list_profiles(sink_id)
        draft["sink"]["profile_name"] = st.selectbox(
            "Sink profile", sink_profiles,
            index=0 if sink_profiles else None, placeholder="Select a saved profile"
        ) if sink_profiles else ""
        draft["sink"]["collection"] = st.text_input("Collection name", draft["sink"].get("collection","KB"))

        st.subheader("Trigger")
        tmode = st.selectbox("Type", ["manual","interval"])
        draft["trigger"]["type"] = tmode
        if tmode == "interval":
            draft["trigger"]["interval_minutes"] = int(st.number_input("Every N minutes", 1, 10080, value=int(draft["trigger"].get("interval_minutes", 60))))

        st.divider()
        cA, cB = st.columns([1,1])
        if cA.button("💾 Save / Update", use_container_width=True):
            pid = None
            if mode == "Edit existing" and pipes:
                pid = sel_id
            else:
                base = re.sub(r"[^A-Za-z0-9_]", "_", draft["name"]).strip("_") or "pipeline"
                pid = base if base not in pipes else f"{base}_{len(pipes)+1}"
            pipes[pid] = draft
            save_pipelines(pipes)
            st.success(f"Saved pipeline `{pid}`.")
            st.rerun()

        if cB.button("▶️ Run now", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    # if creating new but not saved, run the draft temporarily
                    if mode == "Create new":
                        tmp_id = "__tmp_draft__"
                        pipes[tmp_id] = draft
                        save_pipelines(pipes)
                        pid_to_run = tmp_id
                    else:
                        pid_to_run = sel_id
                    result = run_pipeline_by_id(pid_to_run)
                    st.success(f"Ingested {result['ingested']} chunks → **{result['collection']}**.")
                except Exception as e:
                    st.error(f"Run failed: {e}")
                finally:
                    if "__tmp_draft__" in pipes:
                        pipes.pop("__tmp_draft__", None)
                        save_pipelines(pipes)

# ------------------------------ EXAMPLE (import or inline) ----------------------
# Option A (inline): paste all of this into connectors_hub.py and call:
#     render_pipelines_ui()
#
# Option B (import): save as pipelines_inline.py and in connectors_hub.py:
#     from pipelines_inline import render_pipelines_ui
#     render_pipelines_ui()
#
# Trigger loop (external scheduler):
#     from pipelines_inline import trigger_loop_forever
#     trigger_loop_forever(poll_seconds=30)
# --------------------------------------------------------------------------------
