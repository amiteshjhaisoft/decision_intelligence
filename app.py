# Author: Amitesh Jha | iSoft | 2025-10-12 (full, diagnostics, wide file-type support)
# Streamlit RAG chat using Claude + Azure Blob + Weaviate (cloud/local)

from __future__ import annotations

import io, os, json, hashlib, socket, ssl, tempfile, time, re, base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

# ---- IMPORTANT: avoid odd TLS EOFs through proxies/edges (must be set BEFORE importing weaviate)
os.environ.setdefault("HTTPX_DISABLE_HTTP2", "1")

# ---------------------- Optional heavy extractors (graceful if missing) ----------------------
# Images (OCR)
try:
    from PIL import Image, ExifTags  # pip install pillow
except Exception:
    Image, ExifTags = None, None
try:
    import pytesseract  # pip install pytesseract  (requires Tesseract binary installed)
except Exception:
    pytesseract = None

# Audio / Video (ASR)
try:
    import whisper  # pip install -U openai-whisper
except Exception:
    whisper = None
try:
    from moviepy.editor import AudioFileClip  # pip install moviepy
except Exception:
    AudioFileClip = None

# HTML parsing
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

# Office / PDF
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None
try:
    import docx  # pip install python-docx
except Exception:
    docx = None
try:
    from pptx import Presentation  # pip install python-pptx
except Exception:
    Presentation = None
try:
    import pandas as pd  # pip install pandas openpyxl
except Exception:
    pd = None

# Text splitting
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # pip install langchain-text-splitters
except Exception as e:
    raise RuntimeError("Missing dependency: langchain-text-splitters") from e

# Azure + Weaviate + Embeddings + LLM
try:
    from azure.storage.blob import BlobServiceClient
except Exception as e:
    raise RuntimeError("Missing dependency: azure-storage-blob") from e

try:
    import weaviate  # v4 preferred (>=4.6.0)
except Exception as e:
    raise RuntimeError("Missing dependency: weaviate-client") from e

# Detect v4 helpers
try:
    from weaviate.classes.init import Auth
    from weaviate.classes.config import Property, DataType, Configure
    V4 = True
except Exception:
    V4 = False

try:
    from weaviate.classes.tenants import Tenant  # v4 only
except Exception:
    Tenant = None
try:
    from weaviate.classes.query import MetadataQuery
except Exception:
    MetadataQuery = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("Missing dependency: sentence-transformers") from e

try:
    from anthropic import Anthropic
except Exception as e:
    raise RuntimeError("Missing dependency: anthropic") from e


# ---------------------- App constants ----------------------
CLASS_NAME = "KBChunk"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5
# Use a known-good Anthropic model id; change if your account has different access
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"

# Bounds to avoid exploding memory on huge binaries
MAX_BYTES_IMAGE = 25 * 1024 * 1024   # 25 MB
MAX_BYTES_AUDIO = 80 * 1024 * 1024   # 80 MB
MAX_BYTES_VIDEO = 150 * 1024 * 1024  # 150 MB
MAX_BYTES_TEXT  = 50 * 1024 * 1024   # 50 MB

# File type buckets
TEXT_EXTS = {".txt", ".md", ".log"}
CSV_EXT = {".csv"}
JSON_EXT = {".json"}
YAML_EXT = {".yaml", ".yml"}
HTML_EXT = {".html", ".htm"}
DOC_EXT = {".docx"}
PPT_EXT = {".pptx"}
XLS_EXT = {".xlsx"}
PDF_EXT = {".pdf"}
CODE_EXTS = {
    ".py", ".sql", ".js", ".ts", ".java", ".c", ".cpp", ".cs",
    ".go", ".rb", ".rs", ".php", ".xml", ".ini", ".cfg", ".toml"
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ---------------------- Helpers ----------------------
def _hash(s: str) -> str:
    import hashlib as _h
    return _h.md5(s.encode("utf-8")).hexdigest()

def _truncate(s: str, limit: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit] + " ..."

def _safe_decode(b: bytes, encoding="utf-8") -> str:
    try:
        return b.decode(encoding, errors="ignore")
    except Exception:
        return ""

# --------- Parsers (each returns a TEXT representation suitable for RAG) ----------
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

def read_pptx_bytes(b: bytes) -> str:
    if not Presentation:
        return ""
    try:
        prs = Presentation(io.BytesIO(b))
        texts = []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    t = (shape.text or "").strip()
                    if t:
                        slide_text.append(t)
            if slide_text:
                texts.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))
        return "\n\n".join(texts).strip()
    except Exception:
        return ""

def read_xlsx_bytes(b: bytes, limit_rows: int = 5000) -> str:
    if not pd:
        return ""
    try:
        buf = io.BytesIO(b)
        xl = pd.ExcelFile(buf)
        out = []
        for name in xl.sheet_names:
            df = xl.parse(name)
            if len(df) > limit_rows:
                df = df.head(limit_rows)
            out.append(f"[Sheet: {name}]\n" + df.to_csv(index=False))
        return "\n\n".join(out).strip()
    except Exception:
        return ""

def read_csv_bytes(b: bytes, limit_rows: int = 20000) -> str:
    if not pd:
        return _safe_decode(b)
    try:
        buf = io.BytesIO(b)
        df = pd.read_csv(buf, nrows=limit_rows)
        return df.to_csv(index=False)
    except Exception:
        return _safe_decode(b)

def read_html_bytes(b: bytes) -> str:
    if not BeautifulSoup:
        return _safe_decode(b)
    try:
        soup = BeautifulSoup(_safe_decode(b), "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return _safe_decode(b)

def read_yaml_bytes(b: bytes) -> str:
    # no yaml dep; keep raw but pretty
    try:
        s = _safe_decode(b)
        return s.strip()
    except Exception:
        return ""

def read_text_like_bytes(b: bytes, encoding: str = "utf-8") -> str:
    return _safe_decode(b, encoding)

def read_code_bytes(b: bytes) -> str:
    # Just decode; chunker will handle size
    return _safe_decode(b)

def read_image_bytes_ocr(b: bytes, name: str) -> str:
    if not Image:
        return f"[IMAGE: {name}] (Pillow not installed; OCR unavailable)"
    try:
        with Image.open(io.BytesIO(b)) as im:
            # Basic EXIF
            exif_txt = ""
            try:
                exif = getattr(im, "_getexif", lambda: None)()
                if exif:
                    inv = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
                    keep = {k: inv[k] for k in list(inv)[:20]}  # limit output
                    exif_txt = json.dumps(keep, default=str)
            except Exception:
                pass
            # OCR if possible
            ocr_txt = ""
            if pytesseract:
                try:
                    ocr_txt = pytesseract.image_to_string(im) or ""
                except Exception as e:
                    ocr_txt = f"(OCR failed: {e})"
            else:
                ocr_txt = "(OCR skipped: pytesseract not installed)"
            return f"[IMAGE: {name}]\nEXIF: {exif_txt}\nTEXT: {ocr_txt}".strip()
    except Exception as e:
        return f"[IMAGE: {name}] (failed to open: {e})"

def transcribe_audio_bytes(b: bytes, name: str, tmp_dir: Path) -> str:
    if not whisper:
        return f"[AUDIO: {name}] (ASR skipped: whisper not installed)"
    try:
        p = tmp_dir / (Path(name).stem + ".wav")
        p.write_bytes(b)
        model = whisper.load_model("base")
        res = model.transcribe(str(p))
        txt = (res.get("text") or "").strip()
        return f"[AUDIO: {name}]\nTRANSCRIPT: {txt}" if txt else f"[AUDIO: {name}] (no speech detected)"
    except Exception as e:
        return f"[AUDIO: {name}] (ASR failed: {e})"

def transcribe_video_bytes(b: bytes, name: str, tmp_dir: Path) -> str:
    if not whisper or not AudioFileClip:
        return f"[VIDEO: {name}] (ASR skipped: whisper/moviepy not installed)"
    try:
        vid_path = tmp_dir / (Path(name).name.replace("/", "_"))
        aud_path = tmp_dir / (Path(name).stem + ".wav")
        vid_path.write_bytes(b)
        # Extract audio
        clip = AudioFileClip(str(vid_path))
        clip.audio.write_audiofile(str(aud_path), verbose=False, logger=None)
        clip.close()
        # Transcribe
        model = whisper.load_model("base")
        res = model.transcribe(str(aud_path))
        txt = (res.get("text") or "").strip()
        return f"[VIDEO: {name}]\nTRANSCRIPT: {txt}" if txt else f"[VIDEO: {name}] (no speech detected)"
    except Exception as e:
        return f"[VIDEO: {name}] (ASR failed: {e})"

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text or "")

# ---------------------- Azure loader ----------------------
def list_and_download_kb(container: str, prefix: str, connection_string: str, cache_dir: Path) -> List[Tuple[str, Path, int]]:
    """
    Download blobs to cache_dir -> [(blob_name, local_path, size_bytes)]
    Uses ETag cache to avoid re-downloading unchanged blobs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bsc = BlobServiceClient.from_connection_string(connection_string)
    cont = bsc.get_container_client(container)

    results: List[Tuple[str, Path, int]] = []
    blobs = cont.list_blobs(name_starts_with=prefix or "")
    for blob in blobs:
        size = getattr(blob, "size", None)
        if not size or size <= 0:
            continue
        blob_name = blob.name
        local_path = cache_dir / blob_name.replace("/", "__")
        etag_path = cache_dir / (local_path.name + ".etag")

        etag = getattr(blob, "etag", None)
        if local_path.exists() and etag_path.exists() and etag:
            try:
                if etag_path.read_text().strip() == etag:
                    results.append((blob_name, local_path, int(size)))
                    continue
            except Exception:
                pass

        try:
            data = cont.download_blob(blob_name).readall()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            if etag:
                etag_path.write_text(etag)
            results.append((blob_name, local_path, int(size)))
        except Exception as e:
            st.warning(f"Failed to download {blob_name}: {e}")

    return results

# ---------------------- Guess & Read (broad types) ----------------------
def guess_and_read_path(name: str, path: Path, size_bytes: int, tmp_dir: Path) -> str:
    ext = Path(name).suffix.lower()

    # Size guards by category (avoid OOM)
    if ext in IMAGE_EXTS and size_bytes > MAX_BYTES_IMAGE:
        return f"[IMAGE: {name}] (skipped: {size_bytes} bytes > {MAX_BYTES_IMAGE})"
    if ext in AUDIO_EXTS and size_bytes > MAX_BYTES_AUDIO:
        return f"[AUDIO: {name}] (skipped: {size_bytes} bytes > {MAX_BYTES_AUDIO})"
    if ext in VIDEO_EXTS and size_bytes > MAX_BYTES_VIDEO:
        return f"[VIDEO: {name}] (skipped: {size_bytes} bytes > {MAX_BYTES_VIDEO})"
    if (ext in TEXT_EXTS | CSV_EXT | JSON_EXT | YAML_EXT | HTML_EXT | DOC_EXT | PPT_EXT | XLS_EXT | PDF_EXT | CODE_EXTS) and size_bytes > MAX_BYTES_TEXT:
        return f"[TEXT-LIKE: {name}] (skipped: {size_bytes} bytes > {MAX_BYTES_TEXT})"

    b = path.read_bytes()

    # Text-like
    if ext in TEXT_EXTS:
        return read_text_like_bytes(b)
    if ext in CSV_EXT:
        return read_csv_bytes(b)
    if ext in JSON_EXT:
        try:
            j = json.loads(_safe_decode(b))
            return json.dumps(j, indent=2, ensure_ascii=False)
        except Exception:
            return _safe_decode(b)
    if ext in YAML_EXT:
        return read_yaml_bytes(b)
    if ext in HTML_EXT:
        return read_html_bytes(b)
    if ext in DOC_EXT:
        return read_docx_bytes(b)
    if ext in PPT_EXT:
        return read_pptx_bytes(b)
    if ext in XLS_EXT:
        return read_xlsx_bytes(b)
    if ext in PDF_EXT:
        return read_pdf_bytes(b)
    if ext in CODE_EXTS:
        return read_code_bytes(b)

    # Binary media
    if ext in IMAGE_EXTS:
        return read_image_bytes_ocr(b, name)
    if ext in AUDIO_EXTS:
        return transcribe_audio_bytes(b, name, tmp_dir)
    if ext in VIDEO_EXTS:
        return transcribe_video_bytes(b, name, tmp_dir)

    # Fallback
    return _safe_decode(b)

# ---------------------- Connectivity diagnostics ----------------------
def run_weaviate_diagnostics(base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"url": base_url}
    try:
        u = urlparse(base_url)
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "https" else 80)
        info["parsed"] = {"scheme": u.scheme, "host": host, "port": port}

        # DNS
        try:
            addrs = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            info["dns"] = [f"{a[4][0]}:{a[4][1]}" for a in addrs]
        except Exception as e:
            info["dns_error"] = repr(e)

        # TCP
        try:
            s = socket.create_connection((host, port), timeout=6)
            s.close()
            info["tcp_ok"] = True
        except Exception as e:
            info["tcp_error"] = repr(e)

        # HTTPS /.well-known/ready (no auth)
        try:
            import urllib.request
            ready_url = base_url.rstrip("/") + "/v1/.well-known/ready"
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(ready_url, context=ctx, timeout=8) as r:
                info["ready_no_auth"] = f"HTTP {r.status}"
        except Exception as e:
            info["ready_no_auth_error"] = repr(e)

        # HTTPS /.well-known/ready (with Bearer)
        try:
            import urllib.request
            ready_url = base_url.rstrip("/") + "/v1/.well-known/ready"
            ctx = ssl.create_default_context()
            req = urllib.request.Request(ready_url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
                info["ready_with_auth"] = f"HTTP {r.status}"
        except Exception as e:
            info["ready_with_auth_error"] = repr(e)
    except Exception as e:
        info["diagnostics_error"] = repr(e)
    return info

# ---------------------- Weaviate client (v4 with safe fallbacks) ----------------------
def make_weaviate_client(url: str, api_key: Optional[str]) -> Any:
    ver = getattr(weaviate, "__version__", "unknown")
    is_v3_pkg = ver.startswith("3.")
    st.caption(f"📦 weaviate-client version: {ver} (V4 symbols: {bool(V4)})")

    def _parse(u: str):
        if not u or "://" not in u:
            raise ValueError(
                f"Invalid Weaviate URL '{u}'. Use full https URL (e.g., "
                "https://<cluster-id>.weaviate.cloud)."
            )
        p = urlparse(u)
        if not p.hostname:
            raise ValueError(f"Could not parse hostname from '{u}'")
        https = (p.scheme == "https")
        http_port = p.port or (443 if https else 80)
        return p.hostname, https, http_port

    if V4:
        auth = Auth.api_key(api_key) if api_key else None

        # 1) WCS (.weaviate.network / .weaviate.cloud)
        try:
            if (".weaviate.network" in url) or (".weaviate.cloud" in url) or ("wcs" in url):
                st.caption("🔌 connect_to_weaviate_cloud(cluster_url=...)")
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=auth,
                    skip_init_checks=True,
                )
                client.is_connected()
                st.caption("✅ WCS connected")
                return client
        except Exception as e:
            st.warning(f"WCS connect failed (will try custom): {e}")

        # 2) Custom (include grpc args for newer v4)
        try:
            host, https, http_port = _parse(url)
            grpc_host = host
            grpc_secure = https
            grpc_port = 50051
            st.caption(
                f"🔌 connect_to_custom(http={host}:{http_port} https={https}, grpc={grpc_host}:{grpc_port} tls={grpc_secure})"
            )
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=https,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
                auth_credentials=auth,
                skip_init_checks=True,
            )
            client.is_connected()
            st.caption("✅ custom connected")
            return client
        except Exception as e:
            st.warning(f"Custom connect failed (v4): {e}")

    # 3) v3 fallback
    if is_v3_pkg:
        try:
            st.caption("🔌 v3 Client(url=...)")
            from weaviate import Client
            from weaviate.auth import AuthApiKey
            client = Client(url=url, auth_client_secret=AuthApiKey(api_key) if api_key else None)
            client.schema.get()
            st.caption("✅ v3 connected")
            return client
        except Exception as e:
            st.warning(f"v3 fallback failed: {e}")

    raise RuntimeError("All connection strategies failed. Check URL (must include https://), API key, and IP allow-list.")

def ensure_collection(client: Any, class_name: str = CLASS_NAME, tenancy: Optional[str] = None) -> None:
    """
    Ensure a KBChunk collection/class exists. If a tenant name is provided, ensure the tenant exists.
    """
    if V4:
        try:
            existing_names = client.collections.list_all()
            existing_names = [getattr(c, "name", c) for c in existing_names]
        except Exception:
            existing_names = []
        if class_name not in existing_names:
            mt_cfg = Configure.multi_tenancy(enabled=bool(tenancy)) if tenancy else None
            client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                multi_tenancy_config=mt_cfg,
            )
        if tenancy:
            col = client.collections.get(class_name)
            try:
                existing_tenants = [t.name for t in col.tenants.list_all()]
            except Exception:
                existing_tenants = []
            if tenancy not in existing_tenants and Tenant is not None:
                col.tenants.create(Tenant(name=tenancy))
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
    if not items:
        return 0
    if V4:
        col = client.collections.get(class_name, tenant=tenancy) if tenancy else client.collections.get(class_name)
        if tenancy:
            count = 0
            for it, vec in zip(items, vectors):
                col.data.insert(properties=it["properties"], uuid=it["id"], vector=vec)
                count += 1
            return count
        count = 0
        with col.batch.dynamic() as batch:
            for it, vec in zip(items, vectors):
                batch.add_object(properties=it["properties"], uuid=it["id"], vector=vec)
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
        col = client.collections.get(class_name, tenant=tenancy) if tenancy else client.collections.get(class_name)
        meta = MetadataQuery(distance=True) if MetadataQuery else None
        try:
            res = col.query.near_vector(
                vector=query_vec,
                limit=top_k,
                return_properties=["doc_id", "source", "chunk_index", "text"],
                return_metadata=meta,
            )
        except TypeError:
            res = col.query.near_vector(
                near_vector=query_vec,
                limit=top_k,
                return_properties=["doc_id", "source", "chunk_index", "text"],
                return_metadata=meta,
            )
        out: List[Dict[str, Any]] = []
        for o in res.objects:
            props = o.properties or {}
            dist = getattr(getattr(o, "metadata", None), "distance", None)
            out.append({
                "doc_id": props.get("doc_id"),
                "source": props.get("source"),
                "chunk_index": props.get("chunk_index"),
                "text": props.get("text"),
                "score": (1.0 - dist) if isinstance(dist, (float, int)) else None,
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
            {"doc_id": h.get("doc_id"), "source": h.get("source"),
             "chunk_index": h.get("chunk_index"), "text": h.get("text"),
             "score": None}
            for h in hits
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
    tmp_dir = Path(tempfile.gettempdir()) / "kb_tmp_streamlit"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    downloaded = list_and_download_kb(container, prefix, connection_string, cache_dir)
    total = 0
    for blob_name, local_path, size_bytes in downloaded:
        try:
            text = guess_and_read_path(blob_name, local_path, size_bytes, tmp_dir).strip()
            if not text:
                status_cb(f"⚠️ No text extracted from {blob_name}; skipping.")
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
            status_cb(f"Indexed: {blob_name} ({size_bytes} bytes) → {upserted} chunks")
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
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        try:
            return "".join(getattr(b, "text", "") for b in (resp.content or []))
        except Exception:
            return str(resp)

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

    st.divider()
    st.caption("🔎 Connectivity diagnostics")
    if st.button("Run diagnostics"):
        diag = run_weaviate_diagnostics(w_url, w_api_key)
        st.json(diag)

if show_health:
    st.write("**Azure**", {"container": container, "prefix": prefix, "conn_str_set": bool(connection_string)})
    st.write("**Weaviate**", {
        "url": w_url,
        "api_key": bool(w_api_key),
        "client_version": getattr(weaviate, "__version__", "unknown"),
        "v4_mode": V4,
    })
    st.write("**Anthropic**", {"api_key": bool(anthropic_key)})

# Guards
if not w_url:
    st.error("Please set [weaviate].url in .streamlit/secrets.toml (cluster URL, no /v1)")
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
        t = _truncate(t, 2000)
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
    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_holder = st.container()
        try:
            with call_claude(anthropic_key, system_prompt, user_prompt, stream=True) as stream_resp:
                text_accum = ""
                for event in stream_resp:
                    if event.type == "content_block_delta":
                        delta_text = getattr(getattr(event, "delta", None), "text", None)
                        if delta_text:
                            text_accum += delta_text
                            placeholder.markdown(
                                f"<div class='chat-bubble-assistant'>{text_accum}</div>",
                                unsafe_allow_html=True
                            )
                final_text = (text_accum or "").strip()
                if not final_text:
                    final_text = "I couldn't generate a response."
                placeholder.markdown(
                    f"<div class='chat-bubble-assistant'>{final_text}</div>",
                    unsafe_allow_html=True
                )
        except Exception as e:
            final_text = f"Sorry, streaming failed: {e}"
            placeholder.markdown(
                f"<div class='chat-bubble-assistant'>{final_text}</div>",
                unsafe_allow_html=True
            )

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
