# Author: Amitesh Jha | iSoft | 2025-10-12
# rag_claude_weaviate_streamlit.py
# Author: You + ChatGPT
# One-file Streamlit RAG chat using Claude + Azure Blob + Weaviate (cloud/local)
# - Downloads KB files from Azure Blob (prefix), parses common doc types
# - Chunks & embeds locally (MiniLM) and upserts to Weaviate
# - Retrieves top-k matches and sends context to Claude
# - Works with Weaviate v4 or v3 client
#
# Secrets expected in .streamlit/secrets.toml:
# [anthropic]
# api_key = "sk-..."
#
# [azure]
# connection_string = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
# container = "knowledgebase"
# prefix = "KB/"              # optional, can be "" or "KB"
#
# [weaviate]
# url = "https://YOUR-CLUSTER.weaviate.network"
# api_key = "YOUR-WEAVIATE-API-KEY"   # optional if cluster has no auth
# # (Optional) tenancy = "your-tenant"  # only if you use multi-tenancy (v4)
#
# Run: streamlit run rag_claude_weaviate_streamlit.py

from __future__ import annotations

import os, io, re, time, json, hashlib, tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# ---- Core deps
try:
    from azure.storage.blob import BlobServiceClient  # pip install azure-storage-blob
except Exception as e:
    raise RuntimeError("Please install azure-storage-blob") from e

try:
    import weaviate  # pip install weaviate-client
except Exception as e:
    raise RuntimeError("Please install weaviate-client") from e

try:
    # v4 naming
    from weaviate.classes.init import Auth
    from weaviate.classes.config import Property, DataType, Configure
    V4 = True
except Exception:
    V4 = False

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception as e:
    raise RuntimeError("Please install sentence-transformers") from e

try:
    from anthropic import Anthropic  # pip install anthropic
except Exception as e:
    raise RuntimeError("Please install anthropic") from e

# Parsing + chunking
try:
    import pandas as pd  # for quick CSV/tables preview if needed
except Exception:
    pd = None  # optional

try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None

try:
    import docx  # pip install python-docx
except Exception:
    docx = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # pip install langchain-text-splitters
except Exception as e:
    raise RuntimeError("Please install langchain-text-splitters") from e


# ---------------------- Configuration ----------------------
CLASS_NAME = "KBChunk"  # Weaviate collection/class
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5

# ---------------------- Utilities ----------------------
def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def read_pdf_bytes(b: bytes) -> str:
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(io.BytesIO(b))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def read_docx_bytes(b: bytes) -> str:
    if not docx:
        return ""
    try:
        f = io.BytesIO(b)
        d = docx.Document(f)
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""

def read_text_like_bytes(b: bytes, encoding="utf-8") -> str:
    try:
        return b.decode(encoding, errors="ignore")
    except Exception:
        return ""

def guess_and_read_blob(name: str, b: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md", ".log"}:
        return read_text_like_bytes(b)
    if ext in {".csv"}:
        # simple read as text; you may customize to parse columns instead
        return read_text_like_bytes(b)
    if ext in {".json"}:
        try:
            j = json.loads(b.decode("utf-8", "ignore"))
            return json.dumps(j, indent=2, ensure_ascii=False)
        except Exception:
            return read_text_like_bytes(b)
    if ext in {".pdf"}:
        return read_pdf_bytes(b)
    if ext in {".docx"}:
        return read_docx_bytes(b)
    # attempt best-effort text
    return read_text_like_bytes(b)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text or "")

# ---------------------- Azure Loader ----------------------
def list_and_download_kb(container: str, prefix: str, connection_string: str, cache_dir: Path) -> List[Tuple[str, Path]]:
    """
    Downloads blobs under prefix into cache_dir, returns list of (blob_name, local_path).
    Skips empty files. Uses simple ETag-based cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bsc = BlobServiceClient.from_connection_string(connection_string)
    cont = bsc.get_container_client(container)

    results = []
    blobs = cont.list_blobs(name_starts_with=prefix or "")
    for blob in blobs:
        if blob.size is None or blob.size == 0:
            continue
        blob_name = blob.name
        # local path mirrored
        local_path = cache_dir / blob_name.replace("/", "__")
        meta_path = cache_dir / (local_path.name + ".etag")
        # check cache
        etag = getattr(blob, "etag", None)
        if local_path.exists() and meta_path.exists() and etag:
            try:
                if meta_path.read_text().strip() == etag:
                    results.append((blob_name, local_path))
                    continue
            except Exception:
                pass
        # download
        try:
            data = cont.download_blob(blob_name).readall()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            if etag:
                meta_path.write_text(etag)
            results.append((blob_name, local_path))
        except Exception as e:
            st.warning(f"Failed to download {blob_name}: {e}")
    return results

# ---------------------- Weaviate Client (v4 or v3) ----------------------
def make_weaviate_client(url: str, api_key: Optional[str]) -> Any:
    if V4:
        auth = Auth.api_key(api_key) if api_key else None
        try:
            client = weaviate.connect_to_custom(url=url, auth_credentials=auth, skip_init_checks=True)
            return client
        except Exception as e:
            st.error(f"Failed to connect (v4): {e}")
            raise
    else:
        # v3
        try:
            from weaviate import Client
            from weaviate.auth import AuthApiKey
            auth_config = AuthApiKey(api_key) if api_key else None
            client = Client(url=url, auth_client_secret=auth_config)
            return client
        except Exception as e:
            st.error(f"Failed to connect (v3): {e}")
            raise

def ensure_collection(client: Any, class_name: str = CLASS_NAME, tenancy: Optional[str]=None) -> None:
    """
    Ensures a KBChunk collection/class exists.
    For v4: non-vectorizer schema, we'll push vectors manually.
    For v3: same idea.
    """
    if V4:
        cols = client.collections.list_all()
        if class_name in [c.name for c in cols]:
            return
        client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            multi_tenancy_config=Configure.multi_tenancy(enabled=bool(tenancy)) if tenancy else None,
        )
    else:
        # v3
        schema = client.schema.get()
        existing = [c["class"] for c in schema.get("classes", [])]
        if class_name in existing:
            return
        new_class = {
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "source", "dataType": ["text"]},
                {"name": "chunk_index", "dataType": ["int"]},
                {"name": "text", "dataType": ["text"]},
            ],
        }
        client.schema.create_class(new_class)

def upsert_chunks(client: Any, class_name: str, items: List[Dict[str, Any]], vectors: List[List[float]], tenancy: Optional[str]=None) -> int:
    """
    items[i] keys: id (uuid or str), properties dict with doc_id, source, chunk_index, text
    vectors[i] float vector
    """
    if not items:
        return 0

    if V4:
        col = client.collections.get(class_name)
        count = 0
        with col.batch.dynamic() as batch:
            for it, vec in zip(items, vectors):
                props = it["properties"]
                batch.add_object(
                    properties=props,
                    uuid=it["id"],
                    vector=vec,
                    tenant=tenancy,
                )
                count += 1
        return count
    else:
        # v3
        batch = client.batch
        batch.configure(batch_size=64, num_workers=2)
        with batch as b:
            for it, vec in zip(items, vectors):
                props = it["properties"]
                b.add_data_object(
                    data_object=props,
                    class_name=class_name,
                    uuid=it["id"],
                    vector=vec,
                )
        return len(items)

def vector_search(client: Any, class_name: str, query_vec: List[float], top_k: int, tenancy: Optional[str]=None) -> List[Dict[str, Any]]:
    if V4:
        col = client.collections.get(class_name)
        res = col.query.near_vector(
            near_vector=query_vec, limit=top_k, tenant=tenancy,
            return_metadata=["distance"],
            return_properties=["doc_id", "source", "chunk_index", "text"]
        )
        out = []
        for o in res.objects:
            out.append({
                "doc_id": o.properties.get("doc_id"),
                "source": o.properties.get("source"),
                "chunk_index": o.properties.get("chunk_index"),
                "text": o.properties.get("text"),
                "score": 1.0 - (o.metadata.distance or 0.0) if o.metadata and o.metadata.distance is not None else None,
            })
        return out
    else:
        # v3
        res = (
            client.query
                  .get(class_name, ["doc_id", "source", "chunk_index", "text"])
                  .with_near_vector({"vector": query_vec})
                  .with_limit(top_k)
                  .do()
        )
        hits = res.get("data", {}).get("Get", {}).get(class_name, []) or []
        # v3 doesn't return a normalized score here; leave None
        return [
            {
                "doc_id": h.get("doc_id"),
                "source": h.get("source"),
                "chunk_index": h.get("chunk_index"),
                "text": h.get("text"),
                "score": None,
            } for h in hits
        ]

# ---------------------- Embeddings ----------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    # small + decent quality; downloads once
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedder()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
    return vecs

# ---------------------- Ingestion Flow ----------------------
def ingest_from_azure_to_weaviate(connection_string: str, container: str, prefix: str,
                                  w_client: Any, tenancy: Optional[str]=None,
                                  status_cb=lambda s: None) -> int:
    """
    Downloads from Azure, parses, chunks, embeds, upserts to Weaviate.
    Returns number of chunks ingested.
    """
    ensure_collection(w_client, CLASS_NAME, tenancy=tenancy)
    cache_dir = Path(tempfile.gettempdir()) / "kb_cache_streamlit"
    downloaded = list_and_download_kb(container, prefix, connection_string, cache_dir)

    total_chunks = 0
    for blob_name, local_path in downloaded:
        try:
            b = local_path.read_bytes()
            text = guess_and_read_blob(blob_name, b)
            if not text or not text.strip():
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue

            # Build items and vectors
            doc_id = _hash(blob_name)
            props = []
            for i, ch in enumerate(chunks):
                props.append({
                    "id": _hash(f"{doc_id}:{i}"),
                    "properties": {
                        "doc_id": doc_id,
                        "source": blob_name,
                        "chunk_index": i,
                        "text": ch,
                    },
                })
            vectors = embed_texts([p["properties"]["text"] for p in props])
            upserted = upsert_chunks(w_client, CLASS_NAME, props, vectors, tenancy)
            total_chunks += upserted
            status_cb(f"Indexed: {blob_name} → {upserted} chunks")
        except Exception as e:
            status_cb(f"⚠️ Failed on {blob_name}: {e}")
            continue

    return total_chunks

# ---------------------- Anthropic (Claude) ----------------------
def call_claude(api_key: str, system_prompt: str, user_prompt: str, stream: bool=True):
    client = Anthropic(api_key=api_key)
    if stream:
        return client.messages.stream(
            max_tokens=800,
            model="claude-3-7-sonnet-20250219",  # adjust if needed
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.2,
        )
    else:
        return client.messages.create(
            max_tokens=800,
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.2,
        )

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="RAG Chat — Claude + Weaviate + Azure", page_icon="💬", layout="wide")

st.markdown("""
<style>
.chat-bubble-user { background:#e7f0ff; padding:10px 12px; border-radius:14px; }
.chat-bubble-assistant { background:#f7f7f8; padding:10px 12px; border-radius:14px; }
.small-dim { font-size:12px; opacity:.7; }
.source-chip { display:inline-block; font-size:12px; padding:4px 8px; margin:4px 6px 0 0; background:#eef1f4; border-radius:999px; }
</style>
""", unsafe_allow_html=True)

st.title("💬 RAG Chat — Claude + Weaviate + Azure Blob")

# Sidebar config & actions
with st.sidebar:
    st.header("⚙️ Settings")
    az = st.secrets.get("azure", {})
    weav = st.secrets.get("weaviate", {})
    anth = st.secrets.get("anthropic", {})

    # Azure
    connection_string = az.get("connection_string", "")
    container = az.get("container", "")
    prefix = az.get("prefix", "KB/")

    # Weaviate
    w_url = weav.get("url", "")
    w_api_key = weav.get("api_key", None)
    w_tenancy = weav.get("tenancy", None)  # optional

    # Claude
    anthropic_key = anth.get("api_key", "")

    top_k = st.number_input("Top-K passages", min_value=1, max_value=15, value=TOP_K_DEFAULT, step=1)

    st.divider()
    st.caption("🧠 Index KB → Weaviate")
    do_ingest = st.button("Sync / Rebuild Index")

    st.divider()
    st.caption("Health checks")
    show_health = st.checkbox("Show connection details")

if show_health:
    st.write("**Azure**",
             {"container": container, "prefix": prefix, "conn_str_set": bool(connection_string)})
    st.write("**Weaviate**", {"url": w_url, "api_key_set": bool(w_api_key)})
    st.write("**Anthropic**", {"api_key_set": bool(anthropic_key)})

# Connect to Weaviate
if not w_url:
    st.error("Please set [weaviate].url in secrets.toml")
    st.stop()

w_client = make_weaviate_client(w_url, w_api_key)
st.success("✅ Connected to Weaviate")

# Ingestion
if do_ingest:
    if not (connection_string and container):
        st.error("Azure connection_string and container are required in [azure].")
    else:
        status_box = st.empty()
        def _status(msg): status_box.write(msg)
        with st.spinner("Syncing KB from Azure and (re)indexing…"):
            total = ingest_from_azure_to_weaviate(connection_string, container, prefix, w_client, tenancy=w_tenancy, status_cb=_status)
        st.success(f"Done. Upserted {total} chunks.")

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str, "sources": [..]}

# Chat input
st.subheader("Chat")
user_q = st.chat_input("Ask about your Knowledge Base…")
if user_q:
    # 1) Embed question, search Weaviate
    q_vec = embed_texts([user_q])[0]
    hits = vector_search(w_client, CLASS_NAME, q_vec, top_k, tenancy=w_tenancy)

    # 2) Build context
    context_blocks = []
    source_chips = []
    for i, h in enumerate(hits):
        snippet = (h.get("text") or "").strip()
        if not snippet:
            continue
        # small guard to avoid overly long prompts
        snippet = snippet[:2000]
        context_blocks.append(f"[{i+1}] Source: {h.get('source')} (chunk {h.get('chunk_index')})\n{snippet}")
        if h.get("source"):
            source_chips.append(h["source"])

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    system_prompt = (
        "You are a precise assistant that answers ONLY using the provided context when possible. "
        "If the answer is not in the context, say you don't have that info. "
        "Cite sources by their [index] when you use them."
    )
    user_prompt = (
        f"User question:\n{user_q}\n\n"
        f"Context passages (numbered):\n{context_text}\n\n"
        "Instructions:\n- Prefer quoting or paraphrasing the context.\n"
        "- Use [1], [2], ... to cite snippets you relied on.\n"
        "- If important details are missing, say what else you need."
    )

    # 3) Stream to UI
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble-user'>{user_q}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_holder = st.container()
        try:
            with call_claude(anthropic_key, system_prompt, user_prompt, stream=True) as stream_resp:
                text_accum = ""
                for event in stream_resp:
                    if event.type == "message_start":
                        continue
                    if event.type == "content_block_delta":
                        delta = event.delta.get("text", "")
                        if delta:
                            text_accum += delta
                            placeholder.markdown(f"<div class='chat-bubble-assistant'>{text_accum}</div>", unsafe_allow_html=True)
                    elif event.type == "message_delta":
                        continue
                final_text = text_accum.strip()
        except Exception as e:
            final_text = f"Sorry, streaming failed: {e}"
            placeholder.markdown(f"<div class='chat-bubble-assistant'>{final_text}</div>", unsafe_allow_html=True)

        # Show sources
        if source_chips:
            chips_html = "".join([f"<span class='source-chip'>{s}</span>" for s in source_chips[:10]])
            sources_holder.markdown(f"<div class='small-dim'>Sources: {chips_html}</div>", unsafe_allow_html=True)

    # 4) Save to history
    st.session_state.history.append({"role": "user", "content": user_q})
    st.session_state.history.append({"role": "assistant", "content": final_text, "sources": source_chips})

# Render history (above input for continuity on rerun)
if st.session_state.history:
    st.markdown("### Recent conversation")
    for msg in st.session_state.history[-8:]:
        role = msg["role"]
        css = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        with st.chat_message(role):
            st.markdown(f"<div class='{css}'>{msg['content']}</div>", unsafe_allow_html=True)
            if role == "assistant" and msg.get("sources"):
                chips = "".join([f"<span class='source-chip'>{s}</span>" for s in msg["sources"][:10]])
                st.markdown(f"<div class='small-dim'>Sources: {chips}</div>", unsafe_allow_html=True)

# Cleanup client (v4)
if V4:
    try:
        w_client.close()
    except Exception:
        pass
