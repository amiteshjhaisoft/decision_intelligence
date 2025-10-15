# Author: Amitesh Jha | iSoft | 2025-10-12 (full, fixed)
# Streamlit RAG chat using Claude + Azure Blob + Weaviate (cloud/local)
# - Downloads KB from Azure (prefix), parses common doc types
# - Chunks & embeds locally (MiniLM) and upserts to Weaviate
# - Retrieves top-k matches and chats with Claude
# - Works with weaviate-client v4 (WCS/custom) or v3 fallback

from __future__ import annotations

import io, os, json, hashlib, tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

# ------- Required deps (with friendly errors) -------
try:
    from azure.storage.blob import BlobServiceClient  # pip install azure-storage-blob
except Exception as e:
    raise RuntimeError("Missing dependency: azure-storage-blob") from e

try:
    import weaviate  # pip install weaviate-client>=4.6.0
except Exception as e:
    raise RuntimeError("Missing dependency: weaviate-client") from e

# Detect v4 helpers; otherwise we’ll use v3
try:
    from weaviate.classes.init import Auth
    from weaviate.classes.config import Property, DataType, Configure
    V4 = True
except Exception:
    V4 = False

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers") from e

try:
    from anthropic import Anthropic  # pip install anthropic
except Exception as e:
    raise RuntimeError("Missing dependency: anthropic") from e

# Optional parsers
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
    raise RuntimeError("Missing dependency: langchain-text-splitters") from e


# ---------------------- App constants ----------------------
CLASS_NAME = "KBChunk"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"  # stable Claude model


# ---------------------- Helpers ----------------------
def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_pdf_bytes(b: bytes) -> str:
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(io.BytesIO(b))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


def read_docx_bytes(b: bytes) -> str:
    if not docx:
        return ""
    try:
        d = docx.Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""


def read_text_like_bytes(b: bytes, encoding: str = "utf-8") -> str:
    try:
        return b.decode(encoding, errors="ignore")
    except Exception:
        return ""


def guess_and_read_blob(name: str, b: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md", ".log"}:
        return read_text_like_bytes(b)
    if ext == ".csv":
        return read_text_like_bytes(b)
    if ext == ".json":
        try:
            j = json.loads(b.decode("utf-8", "ignore"))
            return json.dumps(j, indent=2, ensure_ascii=False)
        except Exception:
            return read_text_like_bytes(b)
    if ext == ".pdf":
        return read_pdf_bytes(b)
    if ext == ".docx":
        return read_docx_bytes(b)
    # best effort for unknowns
    return read_text_like_bytes(b)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text or "")


# ---------------------- Azure loader ----------------------
def list_and_download_kb(container: str, prefix: str, connection_string: str, cache_dir: Path) -> List[Tuple[str, Path]]:
    """
    Download blobs under prefix into cache_dir, return list of (blob_name, local_path).
    Uses ETag cache to avoid re-downloading unchanged blobs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bsc = BlobServiceClient.from_connection_string(connection_string)
    cont = bsc.get_container_client(container)

    results: List[Tuple[str, Path]] = []
    blobs = cont.list_blobs(name_starts_with=prefix or "")
    for blob in blobs:
        if not getattr(blob, "size", None):
            continue
        blob_name = blob.name
        local_path = cache_dir / blob_name.replace("/", "__")
        etag_path = cache_dir / (local_path.name + ".etag")

        etag = getattr(blob, "etag", None)
        if local_path.exists() and etag_path.exists() and etag:
            try:
                if etag_path.read_text().strip() == etag:
                    results.append((blob_name, local_path))
                    continue
            except Exception:
                pass

        try:
            data = cont.download_blob(blob_name).readall()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            if etag:
                etag_path.write_text(etag)
            results.append((blob_name, local_path))
        except Exception as e:
            st.warning(f"Failed to download {blob_name}: {e}")

    return results


# ---------------------- Weaviate client (v4 / v3) ----------------------
def make_weaviate_client(url: str, api_key: Optional[str]) -> Any:
    """
    Connect to Weaviate:
      - v4 + WCS (.weaviate.network): connect_to_wcs(cluster_url=..., auth_credentials=...)
      - v4 + Self-hosted:            connect_to_custom(http_host, http_port, http_secure, ...)
      - v3 fallback:                 Client(url=..., auth_client_secret=...)
    """
    try:
        st.caption(f"📦 weaviate-client version: {getattr(weaviate, '__version__', 'unknown')}")
    except Exception:
        pass

    # v4 path
    if V4:
        auth = Auth.api_key(api_key) if api_key else None
        try:
            if "weaviate.network" in url or "wcs" in url:
                st.caption("🔌 using connect_to_wcs()")
                return weaviate.connect_to_wcs(
                    cluster_url=url, auth_credentials=auth, skip_init_checks=True
                )

            u = urlparse(url)
            if not u.scheme or not u.netloc:
                raise ValueError(f"Invalid Weaviate URL: {url}")
            secure = (u.scheme == "https")
            host = u.hostname
            port = u.port if u.port else (443 if secure else 80)

            st.caption(f"🔌 using connect_to_custom(): host={host} port={port} secure={secure}")
            return weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                auth_credentials=auth,
                skip_init_checks=True,
            )
        except Exception as e:
            st.error(f"Failed to connect (v4): {e}")
            raise

    # v3 fallback
    try:
        st.caption("🔌 using v3 Client(url=...)")
        from weaviate import Client
        from weaviate.auth import AuthApiKey
        auth_config = AuthApiKey(api_key) if api_key else None
        return Client(url=url, auth_client_secret=auth_config)
    except Exception as e:
        st.error(f"Failed to connect (v3): {e}")
        raise


def ensure_collection(client: Any, class_name: str = CLASS_NAME, tenancy: Optional[str] = None) -> None:
    """
    Ensure a KBChunk collection/class exists (no built-in vectorizer; we push vectors ourselves).
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


def upsert_chunks(client: Any, class_name: str, items: List[Dict[str, Any]], vectors: List[List[float]], tenancy: Optional[str] = None) -> int:
    """
    items[i]: { id: str, properties: {doc_id, source, chunk_index, text} }
    vectors[i]: embedding list
    """
    if not items:
        return 0

    if V4:
        col = client.collections.get(class_name)
        count = 0
        with col.batch.dynamic() as batch:
            for it, vec in zip(items, vectors):
                batch.add_object(
                    properties=it["properties"],
                    uuid=it["id"],
                    vector=vec,
                    tenant=tenancy,
                )
                count += 1
        return count
    else:
        batch = client.batch
        batch.configure(batch_size=64, num_workers=2)
        with batch as b:
            for it, vec in zip(items, vectors):
                b.add_data_object(
                    data_object=it["properties"],
                    class_name=class_name,
                    uuid=it["id"],
                    vector=vec,
                )
        return len(items)


def vector_search(client: Any, class_name: str, query_vec: List[float], top_k: int, tenancy: Optional[str] = None) -> List[Dict[str, Any]]:
    if V4:
        col = client.collections.get(class_name)
        res = col.query.near_vector(
            near_vector=query_vec, limit=top_k, tenant=tenancy,
            return_metadata=["distance"],
            return_properties=["doc_id", "source", "chunk_index", "text"],
        )
        out: List[Dict[str, Any]] = []
        for o in res.objects:
            props = o.properties or {}
            dist = getattr(o, "metadata", None).distance if getattr(o, "metadata", None) else None
            out.append({
                "doc_id": props.get("doc_id"),
                "source": props.get("source"),
                "chunk_index": props.get("chunk_index"),
                "text": props.get("text"),
                "score": 1.0 - dist if dist is not None else None,
            })
        return out
    else:
        res = (
            client.query
            .get(class_name, ["doc_id", "source", "chunk_index", "text"])
            .with_near_vector({"vector": query_vec})
            .with_limit(top_k)
            .do()
        )
        hits = res.get("data", {}).get("Get", {}).get(class_name, []) or []
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
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


# ---------------------- Ingestion ----------------------
def ingest_from_azure_to_weaviate(connection_string: str, container: str, prefix: str,
                                  w_client: Any, tenancy: Optional[str] = None,
                                  status_cb=lambda s: None) -> int:
    ensure_collection(w_client, CLASS_NAME, tenancy=tenancy)
    cache_dir = Path(tempfile.gettempdir()) / "kb_cache_streamlit"
    downloaded = list_and_download_kb(container, prefix, connection_string, cache_dir)

    total = 0
    for blob_name, local_path in downloaded:
        try:
            b = local_path.read_bytes()
            text = guess_and_read_blob(blob_name, b)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue

            doc_id = _hash(blob_name)
            items = [{
                "id": _hash(f"{doc_id}:{i}"),
                "properties": {
                    "doc_id": doc_id,
                    "source": blob_name,
                    "chunk_index": i,
                    "text": ch,
                },
            } for i, ch in enumerate(chunks)]
            vectors = embed_texts([it["properties"]["text"] for it in items])
            upserted = upsert_chunks(w_client, CLASS_NAME, items, vectors, tenancy)
            total += upserted
            status_cb(f"Indexed: {blob_name} → {upserted} chunks")
        except Exception as e:
            status_cb(f"⚠️ Failed on {blob_name}: {e}")
    return total


# ---------------------- Claude ----------------------
def call_claude(api_key: str, system_prompt: str, user_prompt: str, stream: bool = True):
    client = Anthropic(api_key=api_key)
    if stream:
        return client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=800,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    else:
        return client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )


# ---------------------- UI ----------------------
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
    w_tenancy = weav.get("tenancy", None)  # optional (v4 MT)

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
    st.write("**Azure**", {"container": container, "prefix": prefix, "conn_str_set": bool(connection_string)})
    st.write("**Weaviate**", {
        "url": w_url,
        "WEAVIATE_API_KEY": bool(w_api_key),
        "client_version": getattr(weaviate, "__version__", "unknown"),
        "v4_mode": V4,
    })
    st.write("**Anthropic**", {"ANTHROPIC_API_KEY": bool(anthropic_key)})

# Guards
if not w_url:
    st.error("Please set [weaviate].url in .streamlit/secrets.toml")
    st.stop()
if not anthropic_key:
    st.warning("Missing [anthropic].api_key — add your ANTHROPIC key to secrets to chat.")

# Connect
w_client = make_weaviate_client(w_url, w_api_key)
st.success("✅ Connected to Weaviate")

# Ingest
if do_ingest:
    if not (connection_string and container):
        st.error("Azure [connection_string] and [container] are required in secrets.")
    else:
        status_box = st.empty()
        def _status(msg): status_box.write(msg)
        with st.spinner("Syncing KB from Azure and (re)indexing…"):
            total = ingest_from_azure_to_weaviate(connection_string, container, prefix, w_client, tenancy=w_tenancy, status_cb=_status)
        st.success(f"Done. Upserted {total} chunks.")

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Chat")
user_q = st.chat_input("Ask about your Knowledge Base…")

if user_q:
    # 1) embed & retrieve
    q_vec = embed_texts([user_q])[0]
    hits = vector_search(w_client, CLASS_NAME, q_vec, top_k, tenancy=w_tenancy)

    # 2) build prompt context
    context_blocks, source_chips = [], []
    for i, h in enumerate(hits):
        t = (h.get("text") or "").strip()
        if not t:
            continue
        t = t[:2000]
        context_blocks.append(f"[{i+1}] Source: {h.get('source')} (chunk {h.get('chunk_index')})\n{t}")
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

    # 3) stream to UI
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble-user'>{user_q}</div>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_holder = st.container()
        try:
            with call_claude(anthropic_key, system_prompt, user_prompt, stream=True) as stream_resp:
                text_accum = ""
                for event in stream_resp:
                    if event.type == "content_block_delta":
                        delta = event.delta.get("text", "")
                        if delta:
                            text_accum += delta
                            placeholder.markdown(f"<div class='chat-bubble-assistant'>{text_accum}</div>", unsafe_allow_html=True)
                final_text = (text_accum or "").strip()
        except Exception as e:
            final_text = f"Sorry, streaming failed: {e}"
            placeholder.markdown(f"<div class='chat-bubble-assistant'>{final_text}</div>", unsafe_allow_html=True)

        if source_chips:
            chips_html = "".join([f"<span class='source-chip'>{s}</span>" for s in source_chips[:10]])
            sources_holder.markdown(f"<div class='small-dim'>Sources: {chips_html}</div>", unsafe_allow_html=True)

    # 4) save history
    st.session_state.history.append({"role": "user", "content": user_q})
    st.session_state.history.append({"role": "assistant", "content": final_text, "sources": source_chips})

# History
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

# Clean up v4 client
if V4:
    try:
        w_client.close()
    except Exception:
        pass

