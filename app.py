# Author: Amitesh Jha | iSoft | 2025-10-12
# deci_int.py — Streamlit RAG chat (Claude-only, no sidebar, hardcoded settings)
# Milvus rewrite:
# - Vector DB: Milvus / Milvus-Lite (via LangChain Milvus)
# - Chunking: RecursiveCharacterTextSplitter
# - Embeddings: sentence-transformers/all-MiniLM-L6-v2
# Forecast360 customizations preserved:
# - Azure Blob is source of truth (mirror to ./KB even if meta missing)
# - Secrets-first config with diagnostics
# - Auto-index with signature-based change detection
# - Clean chat UI; no sidebar; compact controls

from __future__ import annotations

import os, glob, time, base64, hashlib, json, shutil, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd

# ---- Runtime hygiene
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---- LangChain / Vector (Milvus)
from langchain_community.vectorstores import Milvus
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# ---- Loaders (docx2txt optional; python-docx fallback)
from langchain_community.document_loaders import (
    PyPDFLoader, BSHTMLLoader, CSVLoader, UnstructuredPowerPointLoader
)
try:
    from langchain_community.document_loaders import Docx2txtLoader  # requires docx2txt
    _HAS_DOCX2TXT = True
except Exception:
    _HAS_DOCX2TXT = False
    try:
        from docx import Document as _PyDocxDoc  # from python-docx
    except Exception:
        _PyDocxDoc = None

# ---- Milvus admin / connections
from pymilvus import connections as _milvus_connections, utility as _milvus_utility

# ---- Anthropic (Claude only)
try:
    from anthropic import Anthropic as _AnthropicClientNew
except Exception:
    _AnthropicClientNew = None
try:
    from anthropic import Client as _AnthropicClientOld
except Exception:
    _AnthropicClientOld = None

# ---- Azure SDK (pull KB → ./KB and store index snapshots)
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    try:
        from azure.identity import DefaultAzureCredential
    except Exception:
        DefaultAzureCredential = None  # type: ignore
    _AZURE_OK = True
except Exception:
    BlobServiceClient = ContainerClient = DefaultAzureCredential = None  # type: ignore
    _AZURE_OK = False

# ---------------- Constants
DEFAULT_CLAUDE = "claude-sonnet-4-5"
_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_MODEL_KW = {"device": "cpu", "trust_remote_code": False}
_ENCODE_KW = {"normalize_embeddings": True}

TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".pptx", ".pptm", ".doc", ".odt"}
SPREADSHEET_EXTS = {".xlsx", ".xlsm", ".xltx"}
SUPPORTED_TEXT_DOCS = TEXT_EXTS | DOC_EXTS | SPREADSHEET_EXTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
SUPPORTED_EXTS = SUPPORTED_TEXT_DOCS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

GREETING_RE = re.compile(
    r"""^\s*(hi|hello|hey|hiya|yo|hola|namaste|namaskar|g'day|good\s+(morning|afternoon|evening))[\s!,.?]*$""",
    re.IGNORECASE,
)

# ---------------- Minimal Claude wrapper
class ClaudeDirect(BaseChatModel):
    model: str = DEFAULT_CLAUDE
    temperature: float = 0.2
    max_tokens: int = 800
    _client: object = None

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client)

    @property
    def _llm_type(self) -> str:
        return "claude_direct"

    def _convert_msgs(self, messages: list[BaseMessage]):
        out = []
        for m in messages:
            role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else "user")
            if isinstance(m.content, str):
                text = m.content
            else:
                parts = m.content or []
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parts)
            out.append({"role": role, "content": [{"type": "text", "text": text}]})
        return out

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        amsgs = self._convert_msgs(messages)
        resp = self._client.messages.create(
            model=self.model, messages=amsgs, temperature=self.temperature, max_tokens=self.max_tokens
        )
        text = ""
        for blk in getattr(resp, "content", []) or []:
            if getattr(blk, "type", None) == "text":
                text += getattr(blk, "text", "") or (blk.get("text", "") if isinstance(blk, dict) else "")
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])

# ---------------- Theme / CSS
try:
    st.set_page_config(page_title="Forecast360 • Chat", page_icon="💬", layout="wide")
except Exception:
    pass

USER_AVATAR_PATH: Optional[Path] = None
ASSIST_AVATAR_PATH: Optional[Path] = None

def _first_existing(paths: list[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _resolve_avatar_paths() -> Tuple[Optional[Path], Optional[Path]]:
    user_env = os.getenv("USER_AVATAR_PATH")
    asst_env = os.getenv("ASSISTANT_AVATAR_PATH")
    user = _first_existing([
        Path(user_env).expanduser().resolve() if user_env else None,
        Path.cwd() / "assets" / "avatar.png",
        Path.cwd() / "assets" / "user.png",
        Path.cwd() / "assets" / "me.png",
    ])
    asst = _first_existing([
        Path(asst_env).expanduser().resolve() if asst_env else None,
        Path.cwd() / "assets" / "llm.png",
        Path.cwd() / "assets" / "assistant.png",
        Path.cwd() / "assets" / "bot.png",
        Path.cwd() / "assets" / "robot.png",
    ])
    return user, asst

USER_AVATAR_PATH, ASSIST_AVATAR_PATH = _resolve_avatar_paths()

def _img_to_data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = (path.suffix.lower().lstrip(".") or "png")
    mime = "image/png" if ext in ("png", "apng") else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml")
    return f"data:{mime};base64,{b64}"

USER_AVATAR_URI = _img_to_data_uri(USER_AVATAR_PATH)
ASSIST_AVATAR_URI = _img_to_data_uri(ASSIST_AVATAR_PATH)
user_bg = f"background-image:url('{USER_AVATAR_URI}');" if USER_AVATAR_URI else ""
asst_bg = f"background-image:url('{ASSIST_AVATAR_URI}');" if ASSIST_AVATAR_URI else ""

st.markdown(f"""
<style>
:root{{ --bg:#f7f8fb; --panel:#fff; --text:#0b1220; --border:#e7eaf2;
       --bubble-user:#eef4ff; --bubble-assist:#f6f7fb; --accent:#2563eb; }}
.chat-card{{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
            box-shadow:0 6px 16px rgba(16,24,40,.05); overflow:hidden; }}
.msg{{ display:flex; align-items:flex-start; gap:.65rem; margin:.45rem 0; }}
.avatar{{ width:32px; height:32px; border-radius:50%; border:1px solid var(--border);
          background-size:cover; background-position:center; background-repeat:no-repeat; flex:0 0 32px; }}
.avatar.user {{ {user_bg} }}
.avatar.assistant {{ {asst_bg} }}
.bubble{{ border:1px solid var(--border); background:var(--bubble-assist);
         padding:.8rem .95rem; border-radius:12px; max-width:960px; white-space:pre-wrap; line-height:1.45; }}
.msg.user .bubble{{ background:var(--bubble-user); }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px;
                padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
.small-note{{opacity:.85;font-size:.85rem}}
</style>
""", unsafe_allow_html=True)

# ---------------- KB + Azure helpers

def _local_kb_dir() -> Path:
    p = Path.cwd() / "KB"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _kb_local_version() -> Optional[str]:
    vf = _local_kb_dir() / "meta" / "version.json"
    if vf.exists():
        try:
            return json.loads(vf.read_text(encoding="utf-8")).get("version")
        except Exception:
            return None
    return None

def _azure_cfg() -> Dict[str, Any]:
    try:
        az = st.secrets.get("azure", {})  # type: ignore
    except Exception:
        az = {}
    return {
        "account_url":       az.get("account_url")         or os.getenv("AZURE_ACCOUNT_URL"),
        "connection_string": az.get("connection_string")   or os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        "container":         az.get("container")           or os.getenv("AZURE_BLOB_CONTAINER", "forecast360-kb"),
        "prefix":            az.get("prefix", "KB"),
        "sas_url":           az.get("container_sas_url")   or os.getenv("AZURE_BLOB_CONTAINER_URL"),
    }

def _azure_diag(cfg: Dict[str, Any]) -> str:
    flags = []
    flags.append("sas_url" if cfg.get("sas_url") else "-")
    flags.append("conn_str" if cfg.get("connection_string") else "-")
    flags.append("acct_url" if cfg.get("account_url") else "-")
    return "/".join(flags) + f" · container='{cfg.get('container')}' · prefix='{cfg.get('prefix')}'"

def _azure_container_client() -> Optional["ContainerClient"]:
    if not _AZURE_OK:
        return None
    cfg = _azure_cfg()
    if cfg["sas_url"]:
        try:
            return ContainerClient.from_container_url(cfg["sas_url"])
        except Exception:
            return None
    if cfg["connection_string"]:
        try:
            svc = BlobServiceClient.from_connection_string(cfg["connection_string"])
            return svc.get_container_client(cfg["container"])
        except Exception:
            return None
    if cfg["account_url"] and DefaultAzureCredential is not None:
        try:
            cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            svc = BlobServiceClient(account_url=cfg["account_url"], credential=cred)
            return svc.get_container_client(cfg["container"])
        except Exception:
            return None
    return None

def _azure_kb_version() -> Optional[str]:
    if not _AZURE_OK:
        return None
    try:
        cli = _azure_container_client()
        if not cli:
            return None
        pref = _azure_cfg()["prefix"].rstrip("/") + "/"
        blob = pref + "meta/version.json"
        txt = cli.download_blob(blob).readall().decode("utf-8")
        return json.loads(txt).get("version")
    except Exception:
        return None

def _download_entire_prefix(cli: "ContainerClient", prefix: str, dest: Path) -> int:
    count = 0
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    for blob in cli.list_blobs(name_starts_with=prefix or None):
        rel = blob.name[len(prefix):] if prefix else blob.name
        if not rel or rel.endswith("/"):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = cli.download_blob(blob.name).readall()
            with open(target, "wb") as f:
                f.write(data)
            count += 1
        except Exception:
            pass
    return count

def sync_kb_from_azure_if_needed(status_placeholder=None) -> Tuple[Path, str]:
    kb_root = _local_kb_dir()
    if not _AZURE_OK:
        return kb_root, "Azure SDK missing — using local KB only."

    cli = _azure_container_client()
    if not cli:
        return kb_root, "Azure not configured — using local KB only."

    cfg  = _azure_cfg()
    pref = cfg["prefix"].rstrip("/") if cfg["prefix"] else ""

    remote_ver = _azure_kb_version()
    local_ver  = _kb_local_version()
    local_empty = len(list(kb_root.rglob("*"))) == 0

    should_mirror = (remote_ver is not None and remote_ver != local_ver) or (remote_ver is None and local_empty)

    if not should_mirror:
        label = f"KB sync OK · remote={remote_ver or 'unknown'} · local={local_ver or 'unknown'} · {_azure_diag(cfg)}"
        return kb_root, label

    try:
        shutil.rmtree(kb_root, ignore_errors=True)
    except Exception:
        pass
    kb_root.mkdir(parents=True, exist_ok=True)

    downloaded = _download_entire_prefix(cli, pref, kb_root)
    label = f"Synced {downloaded} files from Azure · remote={remote_ver or 'unknown'} · {_azure_diag(cfg)}"
    return kb_root, label

# ---------------- Milvus connection config + helpers

def _milvus_cfg() -> Dict[str, Any]:
    try:
        mv = st.secrets.get("milvus", {})  # type: ignore
    except Exception:
        mv = {}
    cfg: Dict[str, Any] = {
        "uri":     mv.get("uri") or os.getenv("MILVUS_URI"),
        "host":    mv.get("host") or os.getenv("MILVUS_HOST"),
        "port":    mv.get("port") or os.getenv("MILVUS_PORT"),
        "user":    mv.get("user") or os.getenv("MILVUS_USER"),
        "password":mv.get("password") or os.getenv("MILVUS_PASSWORD"),
        "token":   mv.get("token") or os.getenv("MILVUS_TOKEN"),
        "secure":  mv.get("secure", False) if mv.get("secure") is not None else (os.getenv("MILVUS_SECURE","false").lower()=="true"),
        "db_name": mv.get("db_name") or os.getenv("MILVUS_DB_NAME", "default"),
        "collection": mv.get("collection"),
    }
    # Default to Milvus-Lite local db file next to app if nothing provided
    if not (cfg["uri"] or cfg["host"]):
        cfg["uri"] = str(Path.cwd() / ".milvus" / "milvus.db")
    return cfg

def _milvus_conn_args() -> Dict[str, Any]:
    """
    Build connection args for LangChain Milvus.
    - Milvus-Lite: **plain absolute path** ending with `.db` (no `file:` scheme)
    - Remote Milvus: host/port or full https:// URI + token
    - If user accidentally provides `file:/...`, normalize back to plain path.
    """
    cfg = _milvus_cfg()
    args: Dict[str, Any] = {}
    uri = cfg.get("uri")
    if uri:
        s = str(uri)
        # Strip an accidental file: prefix
        if s.startswith("file:"):
            s = s.replace("file:", "", 1)
        # If it looks like http(s) or another scheme, pass as-is (remote)
        if "://" in s and not s.endswith(".db"):
            args["uri"] = s
        else:
            # Treat as filesystem path for Milvus-Lite
            p = Path(s).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            args["uri"] = str(p)  # PLAIN PATH, e.g. /mount/.../milvus.db
    else:
        # Host/port style (remote Milvus/Zilliz)
        args["host"] = cfg.get("host", "127.0.0.1")
        args["port"] = int(cfg["port"]) if cfg.get("port") else 19530
        if cfg.get("user"):
            args["user"] = cfg["user"]
        if cfg.get("password"):
            args["password"] = cfg["password"]
        if cfg.get("secure"):
            args["secure"] = True
        if cfg.get("token"):
            args["token"] = cfg["token"]
    if cfg.get("db_name"):
        args["db_name"] = cfg["db_name"]
    return args

def _milvus_collection_name(base_kb_dir: str) -> str:
    explicit = _milvus_cfg().get("collection")
    if explicit and isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return f"kb_{hashlib.sha1(base_kb_dir.encode('utf-8')).hexdigest()[:10]}"

def _connect_milvus() -> None:
    """Ensure a default Milvus connection is established once."""
    try:
        _milvus_connections.get_connection_addr("default")
        return
    except Exception:
        pass
    args = _milvus_conn_args()
    try:
        if "uri" in args:
            _milvus_connections.connect(alias="default", uri=args["uri"], token=args.get("token"))
        else:
            _milvus_connections.connect(
                alias="default",
                host=args.get("host", "127.0.0.1"),
                port=args.get("port", 19530),
                user=args.get("user"),
                password=args.get("password"),
                secure=args.get("secure", False),
                token=args.get("token"),
            )
    except Exception as e:
        st.error(f"Milvus connect failed: {e}")

def _milvus_collection_exists(name: str) -> bool:
    try:
        _connect_milvus()
        return _milvus_utility.has_collection(name, using="default")
    except Exception:
        return False

# ---------------- Milvus-Lite <-> Azure Blob snapshot helpers

def _milvus_local_file() -> Optional[Path]:
    """Return the local filesystem path for Milvus-Lite when using a local .db path."""
    args = _milvus_conn_args()
    uri = args.get("uri")
    if not uri:
        return None
    s = str(uri)
    if "://" in s and not s.endswith(".db"):
        return None  # remote Milvus
    return Path(s).expanduser().resolve()

def _azure_index_blob_path() -> str:
    col = st.session_state.get("collection_name", "milvus")
    return f"indexes/{col}.db"  # same container, separate folder

def restore_milvus_db_from_blob_if_exists() -> str:
    if not _AZURE_OK:
        return "Azure SDK not available — skipping Milvus restore."
    local_path = _milvus_local_file()
    if not local_path:
        return "Not using Milvus-Lite (no local .db) — skipping restore."
    cli = _azure_container_client()
    if not cli:
        return "Azure not configured — skipping Milvus restore."
    blob_name = _azure_index_blob_path()
    try:
        props = cli.get_blob_client(blob_name).get_blob_properties()  # raises if missing
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            data = cli.download_blob(blob_name).readall()
            f.write(data)
        return f"Restored Milvus DB from Azure blob ‘{blob_name}’ ({props.size} bytes)."
    except Exception:
        return "No Milvus snapshot found in Azure — fresh index will be created."

def backup_milvus_db_to_blob() -> str:
    if not _AZURE_OK:
        return "Azure SDK not available — skipping Milvus backup."
    local_path = _milvus_local_file()
    if not local_path or not local_path.exists():
        return "No local Milvus DB file to back up — skipping."
    cli = _azure_container_client()
    if not cli:
        return "Azure not configured — skipping Milvus backup."
    blob_name = _azure_index_blob_path()
    try:
        with open(local_path, "rb") as f:
            cli.upload_blob(name=blob_name, data=f, overwrite=True)
        return f"Backed up Milvus DB to Azure blob ‘{blob_name}’."
    except Exception as e:
        return f"Milvus backup failed: {e}"

# ---------------- Utils

def human_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.2f} s"

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def iter_files(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(list(set(paths)))

def compute_kb_signature(folder: str) -> Tuple[str, int]:
    files = iter_files(folder)
    lines = []
    base = os.path.abspath(folder)
    for p in files:
        try:
            stt = os.stat(p)
            rel = os.path.relpath(os.path.abspath(p), base)
            lines.append(f"{rel}|{stt.st_size}|{int(stt.st_mtime)}")
        except Exception:
            continue
    lines.sort()
    raw = "\n".join(lines) + str(SUPPORTED_TEXT_DOCS)
    return stable_hash(raw if raw else f"EMPTY-{time.time()}"), len(files)

# ---------------- Loading

def _fallback_read(path: str) -> str:
    try:
        if path.lower().endswith(tuple(SPREADSHEET_EXTS)):
            df = pd.read_excel(path).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"Spreadsheet content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        if path.lower().endswith((".csv", ".tsv")):
            sep = "\t" if path.lower().endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"CSV/TSV content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading file {Path(path).name}: {e}")
        return ""

def load_one(path: str) -> List[Document]:
    p = path.lower()
    if p.endswith(tuple(IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS)):
        doc_type = "Image" if p.endswith(tuple(IMAGE_EXTS)) else ("Audio" if p.endswith(tuple(AUDIO_EXTS)) else "Video")
        placeholder_content = (
            f"This document is a {doc_type} file. "
            f"Text content unavailable (requires OCR/transcription). "
            f"Metadata: {Path(path).name}."
        )
        return [Document(page_content=placeholder_content, metadata={"source": path, "type": doc_type, "status": "placeholder"})]

    try:
        if p.endswith(".pdf"):
            return PyPDFLoader(path).load()
        if p.endswith((".html", ".htm")):
            return BSHTMLLoader(path).load()
        if p.endswith(".docx"):
            if _HAS_DOCX2TXT:
                return Docx2txtLoader(path).load()
            try:
                if '._PyDocxDoc' in globals() and _PyDocxDoc is not None:
                    d = _PyDocxDoc(path)
                    text = "\n".join([para.text for para in d.paragraphs]) or ""
                    return [Document(page_content=text, metadata={"source": path})] if text.strip() else []
            except Exception as e:
                st.warning(f"python-docx fallback failed for {Path(path).name}: {e}")
                return []
        if p.endswith((".pptx", ".pptm")):
            return UnstructuredPowerPointLoader(path).load()
        if p.endswith(".csv"):
            return CSVLoader(path).load()
        if p.endswith(".tsv"):
            return CSVLoader(path, csv_args={"delimiter": "\t"}).load()
        if p.endswith(tuple(TEXT_EXTS | SPREADSHEET_EXTS | {".doc", ".odt"})):
            txt = _fallback_read(path)
            return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
        txt = _fallback_read(path)
        return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
    except Exception as e:
        st.warning(f"Failed to load/process {Path(path).name}. Error: {e}")
        return []

def load_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    files_to_load = [p for p in iter_files(folder) if Path(p).suffix.lower() in SUPPORTED_EXTS]
    for path in files_to_load:
        docs.extend(load_one(path))
    return docs

# ---------------- Embeddings

@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200

@st.cache_resource(show_spinner=False)
def _cached_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=_EMB_MODEL,
        model_kwargs=_EMB_MODEL_KW,
        encode_kwargs=_ENCODE_KW,
    )

def _make_embeddings():
    return _cached_embeddings()

# ---------------- Milvus indexing & retrieval

def _split_docs(docs: List[Document], cfg: ChunkingConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)

def index_folder_milvus(folder: str, collection_name: str, chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    """
    (re)Build Milvus collection with fresh chunks from the local KB folder
    (which was mirrored from Azure Blob). Returns (#raw_docs, #chunks).
    """
    raw_docs = load_documents(folder)
    if not raw_docs:
        try:
            if _milvus_collection_exists(collection_name):
                _milvus_utility.drop_collection(collection_name, using="default")
        except Exception:
            pass
        return (0, 0)

    chunks = _split_docs(raw_docs, chunk_cfg)
    embeddings = _make_embeddings()

    _connect_milvus()
    conn_args = _milvus_conn_args()

    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=conn_args,
        drop_old=True,   # idempotent rebuilds on change
    )

    # After successful (re)index, back up Milvus-Lite DB to Azure (if configured)
    try:
        note = backup_milvus_db_to_blob()
        st.markdown(f'<div class="status-inline">{note}</div>', unsafe_allow_html=True)
    except Exception:
        pass

    return (len(raw_docs), len(chunks))

def get_vectorstore(collection_name: str) -> Optional[Milvus]:
    try:
        _connect_milvus()
        if not _milvus_collection_exists(collection_name):
            return None
        conn_args = _milvus_conn_args()
        return Milvus(
            embedding_function=_make_embeddings(),
            collection_name=collection_name,
            connection_args=conn_args,
        )
    except Exception as e:
        st.error(f"Failed to open Milvus collection '{collection_name}': {e}")
        return None

# ---------------- Claude init

def _strip_proxy_env() -> None:
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(v, None)

def _get_secret_api_key() -> Optional[str]:
    try:
        s = st.secrets
    except Exception:
        s = None
    if s:
        for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for parent in ("anthropic","claude","secrets"):
            if parent in s and isinstance(s[parent], dict):
                ns = s[parent]
                for k in ("api_key","ANTHROPIC_API_KEY","key","token"):
                    v = ns.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

@st.cache_resource(show_spinner=False)
def _anthropic_client_from_secrets_cached():
    _strip_proxy_env()
    api_key = _get_secret_api_key()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY. Put it in .streamlit/secrets.toml under [anthropic] api_key=\"...\" or env.")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if _AnthropicClientNew is not None:
        return _AnthropicClientNew(api_key=api_key)
    if _AnthropicClientOld is not None:
        return _AnthropicClientOld(api_key=api_key)
    raise RuntimeError("Anthropic SDK not installed correctly.")

def make_llm(model_name: str, temperature: float):
    client = _anthropic_client_from_secrets_cached()
    return ClaudeDirect(client=client, model=model_name or DEFAULT_CLAUDE,
                        temperature=temperature, max_tokens=800)

def make_chain(vs: Milvus, llm: BaseChatModel, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False
    )

# ---------------- Defaults + auto-index

def settings_defaults() -> Dict[str, Any]:
    kb_dir = str(_local_kb_dir())
    return {
        "base_folder": kb_dir,
        "collection_name": _milvus_collection_name(kb_dir),
        "emb_model": _EMB_MODEL,
        "chunk_cfg": ChunkingConfig(),
        "claude_model": DEFAULT_CLAUDE,
        "temperature": 0.2,
        "top_k": 5,
        "auto_index_min_interval_sec": 8,
        "chat_height": 560,
    }

def auto_index_if_needed(status_placeholder: Optional[object] = None) -> Optional[Milvus]:
    folder   = st.session_state.get("base_folder")
    colname  = st.session_state.get("collection_name")
    min_gap  = int(st.session_state.get("auto_index_min_interval_sec", 8))
    target   = status_placeholder if status_placeholder is not None else st

    sig_now, file_count = compute_kb_signature(folder)
    last_sig  = st.session_state.get("_kb_last_sig")
    last_time = float(st.session_state.get("_kb_last_index_ts", 0.0))
    now       = time.time()

    need_index = (last_sig != sig_now) or (last_sig is None)
    throttled  = (now - last_time) < min_gap

    vs = get_vectorstore(colname)
    coll_exists = vs is not None

    if (not coll_exists) or (need_index and not throttled):
        try:
            target.markdown('<div class="status-inline">Indexing into Milvus…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_milvus(folder, colname, st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"]      = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"]   = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            label = f"Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks"
        except Exception as e:
            label = f"Auto-index failed: <b>{e}</b>"
        target.markdown(f'<div class="status-inline">{label}</div>', unsafe_allow_html=True)
        vs = get_vectorstore(colname)  # reopen after (re)build
    else:
        ts = st.session_state.get("_kb_last_index_ts")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        target.markdown(
            f'<div class="status-inline">Auto-index <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Collection: <code>{colname}</code></div>',
            unsafe_allow_html=True
        )

    return vs

# ---------------- UI helpers

def _avatar_for_role(role: str) -> Optional[str]:
    if role == "user":
        return USER_AVATAR_URI or (str(USER_AVATAR_PATH) if USER_AVATAR_PATH else "👤")
    if role == "assistant":
        return ASSIST_AVATAR_URI or (str(ASSIST_AVATAR_PATH) if ASSIST_AVATAR_PATH else "🤖")
    return None

def render_chat_history():
    for message in st.session_state.get("messages", []):
        role = message["role"]
        with st.chat_message(role, avatar=_avatar_for_role(role)):
            st.markdown(message["content"])

def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    if not source_docs:
        return ""
    from collections import Counter
    names = []
    for d in source_docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "unknown")
        try:
            display = str(Path(src).resolve().relative_to(Path(kb_root).resolve())) if kb_root else Path(src).name
        except Exception:
            display = Path(src).name
        names.append(display)
    counts = Counter(names)
    lines = [f"- {name}" + (f" ×{n}" if n > 1 else "") for name, n in counts.items()]
    return "\n\n**Sources**\n" + "\n".join(lines)

def read_whole_file_from_disk(path: str) -> str:
    docs = load_one(path)
    return "".join(
        (f"\n\n--- [{i}] {Path((d.metadata or {}).get('source','')).name} ---\n") + (d.page_content or "")
        for i, d in enumerate(docs, 1)
    ).strip()

def read_whole_doc_by_name(name_or_stem: str, base_folder: str) -> Tuple[str, List[str]]:
    name_or_stem = name_or_stem.lower().strip()
    candidates = [p for p in iter_files(base_folder) if name_or_stem in os.path.basename(p).lower()]
    texts = []
    for p in candidates:
        try:
            texts.append(read_whole_file_from_disk(p))
        except Exception as e:
            texts.append(f"[Error reading {os.path.basename(p)}: {e}]")
    return ("\n\n".join(t for t in texts if t.strip()) or ""), candidates

# ---------------- LLM & Chain

def make_llm_and_chain(vs: Milvus):
    llm = make_llm(st.session_state["claude_model"], float(st.session_state["temperature"]))
    chain = make_chain(vs, llm, int(st.session_state["top_k"]))
    return llm, chain

_SLASH_SET_RE = re.compile(r"/set\s+(.+)$", re.IGNORECASE)
_KV_RE = re.compile(r"(\w+)\s*=\s*([\w\.\-]+)")

def _apply_settings_kv(s: str) -> str:
    changed = []
    for k, v in _KV_RE.findall(s):
        try:
            if k in {"top_k", "chat_height"}:
                st.session_state[k] = int(v)
            elif k in {"temperature"}:
                st.session_state[k] = float(v)
            elif k in {"claude_model"}:
                st.session_state[k] = v
            elif k in {"chunk_size", "chunk_overlap"}:
                cfg = st.session_state.get("chunk_cfg", ChunkingConfig())
                if k == "chunk_size":
                    cfg.chunk_size = int(v)
                else:
                    cfg.chunk_overlap = int(v)
                st.session_state["chunk_cfg"] = cfg
            changed.append(f"{k}→{st.session_state[k]}")
        except Exception:
            pass
    return ", ".join(changed) if changed else "No changes applied."

def handle_user_input(query: str, vs: Optional[Milvus]):
    st.session_state.setdefault("messages", [])

    mset = _SLASH_SET_RE.search(query)
    if mset:
        applied = _apply_settings_kv(mset.group(1))
        st.session_state["messages"].append({"role": "assistant", "content": f"Applied settings: {applied}"})
        st.rerun(); return

    st.session_state["messages"].append({"role": "user", "content": query})

    m = re.match(r"^\s*(read|open|show)\s+(.+)$", query, flags=re.IGNORECASE)
    if m:
        target = m.group(2).strip().strip('"').strip("'")
        full_text, files = read_whole_doc_by_name(target, st.session_state["base_folder"])
        if not files:
            st.session_state["messages"].append({"role": "assistant", "content": f"Couldn't find a file containing “{target}” in the Knowledge Base folder."})
            st.rerun(); return

        if len(full_text) > 8000:
            try:
                llm, _ = make_llm_and_chain(vs or Milvus(
                    embedding_function=_make_embeddings(),
                    collection_name=st.session_state["collection_name"],
                    connection_args=_milvus_conn_args()))
                summary = llm.predict(f"Summarize the following document concisely, focusing on key facts and numbers:\n\n{full_text[:180000]}")
                reply = f"**Full-document summary for:** {', '.join(Path(p).name for p in files)}\n\n{summary}"
            except Exception as e:
                reply = f"Loaded the full document but failed to summarize: {e}\n\n--- RAW BEGIN ---\n{full_text[:20000]}\n--- RAW TRUNCATED ---"
        else:
            reply = f"**Full document content:**\n\n{full_text}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun(); return

    if GREETING_RE.match(query):
        st.session_state["messages"].append({"role": "assistant", "content": "Hello"})
        st.rerun(); return

    if vs is None:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "I couldn’t open a Milvus collection yet. Make sure your Azure KB has at least one readable text document (PDF, DOCX, CSV, etc.). I’ll auto-index as soon as files are available."
        })
        st.rerun(); return

    t0 = time.time()
    try:
        _, chain = make_llm_and_chain(vs)
        with st.spinner("Querying Claude with RAG..."):
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "Not found in Knowledge Base."
            sources = result.get("source_documents", []) or []
        citation_block = build_citation_block(sources, kb_root=st.session_state.get("base_folder"))
        msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
    except Exception as e:
        msg = f"RAG error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.rerun()

# ---------------- Main

def main():
    # 1) Defaults
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

    # 2) Sync KB from Azure (files that will be chunked & embedded)
    kb_dir, sync_label = sync_kb_from_azure_if_needed()
    st.session_state["base_folder"] = str(kb_dir)
    # Recompute collection name if not explicitly provided
    if not _milvus_cfg().get("collection"):
        st.session_state["collection_name"] = _milvus_collection_name(str(kb_dir))

    # Try to restore a previous Milvus-Lite snapshot from the SAME blob container
    restore_note = restore_milvus_db_from_blob_if_exists()
    st.markdown(f"<div class='status-inline'>{restore_note}</div>", unsafe_allow_html=True)

    # 3) Header + controls
    st.markdown("### 💬 Chat with Forecast360")
    st.markdown(f"<div class='status-inline'><b>KB Sync:</b> {sync_label}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1], vertical_alignment="center")
    with c1:
        hero_status = st.container()
        vs = auto_index_if_needed(status_placeholder=hero_status)
    with c2:
        if st.button("Reindex KB", use_container_width=True):
            st.session_state.pop("_kb_last_sig", None)
            st.session_state["_kb_last_index_ts"] = 0
            vs = auto_index_if_needed(status_placeholder=hero_status)
    with c3:
        st.session_state["chat_height"] = st.number_input("Chat height (px)", 420, 1200, st.session_state.get("chat_height", 560), step=20)

    # 4) Chat UI
    st.session_state.setdefault("messages", [{"role": "assistant", "content": "Hi! Ask me anything about Forecast360."}])
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    chat_area = st.container(height=int(st.session_state.get("chat_height", 560)), border=False)
    with chat_area:
        render_chat_history()
    user_text = st.chat_input("Type your question… ")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)

if __name__ == "__main__":
    main()
