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

# ------------------------------ UI (3) ------------------------------------------
# This section is safe to call inline from Streamlit-enabled connectors_hub.py

def render_pipelines_ui():
    import json as _json
    try:
        import streamlit as st
    except Exception:
        raise RuntimeError("Streamlit is required to render the pipelines UI.")

    # ---------------- state ----------------
    st.session_state.setdefault("pipelines_show_editor", False)
    st.session_state.setdefault("pipelines_edit_id", None)
    st.session_state.setdefault("pipelines_draft", None)

    def _new_empty_draft() -> Dict[str, Any]:
        # No defaults for source/sink — empty until user picks from saved connections
        return {
            "name": "",
            "source": {
                "connector_id": "",   # selected from saved connections
                "profile_name": "",   # selected from saved profiles
                # optional per-pipeline params (user may fill if needed)
                "container": "", "prefix": "", "pattern": r".*",
                "root": "", "name": ""
            },
            "chunking": {"method":"recursive","chunk_size":800,"chunk_overlap":80},
            "embedding": {"model":"sentence-transformers/all-MiniLM-L6-v2","batch_size":64},
            "sink": {
                "connector_id":"",    # selected from saved connections
                "profile_name":"",    # selected from saved profiles
                "collection":""
            },
            "trigger": {"type":"manual","interval_minutes": 0}  # manual only (UI enforces)
        }

    def _open_new():
        st.session_state.pipelines_edit_id = None
        st.session_state.pipelines_draft = _new_empty_draft()
        st.session_state.pipelines_show_editor = True

    def _open_edit(pid: str, payload: Dict[str, Any]):
        st.session_state.pipelines_edit_id = pid
        st.session_state.pipelines_draft = _json.loads(_json.dumps(payload))
        st.session_state.pipelines_show_editor = True

    def _close_editor():
        st.session_state.pipelines_show_editor = False
        st.session_state.pipelines_edit_id = None
        st.session_state.pipelines_draft = None

    # ---------------- stores ----------------
    conns = load_connections()
    pipes = load_pipelines()

    # Which connectors are actually available from saved connections?
    saved_connector_ids = sorted(conns.keys())

    # Only allow sources/sinks that the runner supports *and* exist in saved connections
    supported_sources = {"azureblob", "localfs"}
    supported_sinks   = {"weaviate"}

    available_sources = [c for c in saved_connector_ids if c in supported_sources]
    available_sinks   = [c for c in saved_connector_ids if c in supported_sinks]

    def _profiles_for(conn_id: str) -> List[str]:
        return sorted(list((conns.get(conn_id) or {}).keys()))

    # ---------------- UI ----------------
    st.markdown("## 🛠️ Pipelines")
    st.caption("Design pipelines that extract → chunk → embed → sync to your Vector DB (ADF-style).")

    hdr_l, hdr_r = st.columns([1,5])
    with hdr_l:
        if st.button("➕ Create pipeline", type="primary", use_container_width=True):
            _open_new()

    left, right = st.columns([4,6], gap="large")

    # -------- left: saved list --------
    with left:
        st.markdown("### 📜 Saved pipelines")
        if not pipes:
            if not available_sources or not available_sinks:
                st.warning("No pipelines yet. Also ensure you’ve saved at least one **source** and one **vector DB (sink)** connection in the Connections hub.")
            else:
                st.info("No pipelines yet. Click **Create pipeline** to add one.")
        else:
            for pid, meta in sorted(pipes.items(), key=lambda kv: kv[1].get("name","").lower()):
                c = st.container(border=True)
                with c:
                    st.markdown(f"**{meta.get('name','(unnamed)')}**  `{pid}`")
                    a, b, d = st.columns([2,2,2])
                    if a.button("▶️ Run", key=f"run::{pid}", use_container_width=True):
                        with st.spinner("Running..."):
                            try:
                                result = run_pipeline_by_id(pid)
                                st.success(f"Ingested {result['ingested']} chunks → **{result['collection']}**.")
                            except Exception as e:
                                st.error(f"Run failed: {e}")
                    if b.button("✏️ Edit", key=f"edit::{pid}", use_container_width=True):
                        _open_edit(pid, meta)
                        st.rerun()
                    if d.button("🗑️ Delete", key=f"del::{pid}", use_container_width=True):
                        try:
                            pipes.pop(pid, None)
                            save_pipelines(pipes)
                            st.toast("Pipeline deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

    # -------- right: editor --------
    with right:
        if not st.session_state.pipelines_show_editor:
            st.markdown("### ✏️ Create / Edit pipeline")
            st.info("Click **Create pipeline** or **Edit** on a saved item to open the editor.")
            return

        draft = st.session_state.pipelines_draft or _new_empty_draft()
        editing_id = st.session_state.pipelines_edit_id
        title = "Create pipeline" if editing_id is None else f"Edit pipeline — `{editing_id}`"
        st.markdown(f"### ✏️ {title}")

        # ---- Basic
        draft["name"] = st.text_input("Pipeline name", draft.get("name",""))

        st.divider()
        # ---- Source (no defaults; from saved connections only)
        st.subheader("Source (from saved connections)")
        src_conn = st.selectbox(
            "Source connector",
            available_sources,
            index=None,
            placeholder="Select a saved source connector",
        )
        if src_conn is None:
            draft["source"]["connector_id"] = ""
            src_profiles = []
        else:
            draft["source"]["connector_id"] = src_conn
            src_profiles = _profiles_for(src_conn)

        src_prof = st.selectbox(
            "Source profile",
            src_profiles,
            index=None,
            placeholder="Select a saved source profile",
        ) if src_profiles else None
        draft["source"]["profile_name"] = src_prof or ""

        # Optional per-pipeline params (keep empty unless the user wants overrides)
        if src_conn == "azureblob":
            draft["source"]["container"] = st.text_input("Container (optional)", draft["source"].get("container",""))
            draft["source"]["prefix"] = st.text_input("Prefix (optional)", draft["source"].get("prefix",""))
            draft["source"]["pattern"] = st.text_input("Filename regex", draft["source"].get("pattern", r".*"))
        elif src_conn == "localfs":
            draft["source"]["root"] = st.text_input("Local folder path (optional)", draft["source"].get("root",""))
            draft["source"]["pattern"] = st.text_input("Filename regex", draft["source"].get("pattern", r".*"))

        draft["source"]["name"] = st.text_input("Source label (optional)", draft["source"].get("name",""))

        # ---- Chunking
        st.subheader("Chunking")
        c1, c2, c3 = st.columns(3)
        draft["chunking"]["method"] = c1.selectbox("Method", ["recursive","tokens"], index=0 if draft["chunking"].get("method","recursive")=="recursive" else 1)
        draft["chunking"]["chunk_size"] = int(c2.number_input("Chunk size", 100, 5000, value=int(draft["chunking"].get("chunk_size",800)), step=50))
        draft["chunking"]["chunk_overlap"] = int(c3.number_input("Overlap", 0, 1000, value=int(draft["chunking"].get("chunk_overlap",80)), step=10))

        # ---- Embedding
        st.subheader("Embedding")
        e1, e2 = st.columns([3,1])
        draft["embedding"]["model"] = e1.text_input("Sentence-Transformers model", draft["embedding"].get("model","sentence-transformers/all-MiniLM-L6-v2"))
        draft["embedding"]["batch_size"] = int(e2.number_input("Batch size", 1, 4096, value=int(draft["embedding"].get("batch_size",64)), step=1))

        # ---- Sink (Sync) — from saved connections only; no defaults
        st.subheader("Sync (Vector DB) — from saved connections")
        sink_conn = st.selectbox(
            "Sink connector",
            available_sinks,
            index=None,
            placeholder="Select a saved sink connector",
        )
        if sink_conn is None:
            draft["sink"]["connector_id"] = ""
            sink_profiles = []
        else:
            draft["sink"]["connector_id"] = sink_conn
            sink_profiles = _profiles_for(sink_conn)

        sink_prof = st.selectbox(
            "Sink profile",
            sink_profiles,
            index=None,
            placeholder="Select a saved sink profile",
        ) if sink_profiles else None
        draft["sink"]["profile_name"] = sink_prof or ""

        draft["sink"]["collection"] = st.text_input("Collection name (optional)", draft["sink"].get("collection",""))

        # ---- Trigger (manual only)
        st.subheader("Trigger")
        st.selectbox("Type", ["manual"], index=0, disabled=True)
        draft["trigger"]["type"] = "manual"
        draft["trigger"]["interval_minutes"] = 0

        st.divider()

        # Validation: must pick source connector/profile and sink connector/profile
        has_valid_source = bool(draft["source"]["connector_id"] and draft["source"]["profile_name"])
        has_valid_sink   = bool(draft["sink"]["connector_id"] and draft["sink"]["profile_name"])

        warn_msg = []
        if not available_sources:
            warn_msg.append("No saved **source** connections found.")
        if not available_sinks:
            warn_msg.append("No saved **sink (vector DB)** connections found.")
        if not has_valid_source:
            warn_msg.append("Select a **Source connector** and **Source profile**.")
        if not has_valid_sink:
            warn_msg.append("Select a **Sink connector** and **Sink profile**.")
        if warn_msg:
            st.warning(" ".join(warn_msg))

        btn_save, btn_run, btn_cancel = st.columns([1,1,1])
        if btn_save.button("💾 Save", use_container_width=True, disabled=not (has_valid_source and has_valid_sink)):
            # Assign ID
            pid = st.session_state.pipelines_edit_id
            if pid is None:
                base = re.sub(r"[^A-Za-z0-9_]", "_", (draft.get("name","") or "pipeline").strip()) or "pipeline"
                suffix = 1
                new_id = base
                while new_id in pipes:
                    suffix += 1
                    new_id = f"{base}_{suffix}"
                pid = new_id
            pipes[pid] = draft
            save_pipelines(pipes)
            st.toast("Pipeline saved.")
            _close_editor()
            st.rerun()

        if btn_run.button("▶️ Run now", use_container_width=True, disabled=not (has_valid_source and has_valid_sink)):
            # Run a temporary draft without saving
            temp_id = "__tmp_draft__"
            pipes[temp_id] = draft
            save_pipelines(pipes)
            try:
                with st.spinner("Running..."):
                    result = run_pipeline_by_id(temp_id)
                st.success(f"Ingested {result['ingested']} chunks → **{result['collection']}**.")
            except Exception as e:
                st.error(f"Run failed: {e}")
            finally:
                pipes.pop(temp_id, None)
                save_pipelines(pipes)

        if btn_cancel.button("✖️ Cancel", use_container_width=True):
            _close_editor()
            st.rerun()

# ------------------------------ EXAMPLE
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
