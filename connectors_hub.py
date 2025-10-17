# Author: Amitesh Jha | iSoft | 2025-10-12
# connectors_hub.py
# Professional Connectors Hub (Streamlit) — hardened storage, schema, ENV secrets, and pipeline guards

from __future__ import annotations

import base64
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

APP_TITLE = "🔌 Data Connectors Hub"
APP_TAGLINE = "Configure, validate, and organize connection profiles for databases, warehouses, NoSQL, storage, streaming, and SaaS."

# Resolve paths robustly (works on local + Streamlit Cloud)
try:
    APP_DIR = Path(__file__).parent
except NameError:
    APP_DIR = Path.cwd()

CONN_STORE = APP_DIR / "connections.json"
PIPE_STORE = APP_DIR / "pipelines.json"
ASSETS_DIR = APP_DIR / "assets"  # optional logo files

# ---------------------- Page Config & Global Style ----------------------
st.set_page_config(page_title="Connectors Hub", layout="wide", page_icon="🔌")

st.markdown(
    """
    <style>
      .main > div { padding-top: 1rem; }
      .app-title { font-size: 1.9rem; font-weight: 700; letter-spacing: .2px; }
      .app-tag { color: #6b7280; margin-top: .35rem; }
      .card {
        border: 1px solid #E5E7EB; border-radius: 10px; padding: 1rem 1.1rem;
        background: #FFFFFF; box-shadow: 0 1px 2px rgba(0,0,0,.04);
      }
      .card h3, .card h4 { margin: 0 0 .6rem 0; }
      .muted { color: #6b7280; }
      .small { font-size: .92rem; }
      .kpi {
        display: inline-flex; align-items: center; gap:.5rem; padding:.45rem .7rem;
        border:1px solid #E5E7EB; border-radius: 999px; background:#F9FAFB; margin-right: .5rem;
      }
      .pill { display:inline-block; padding:.18rem .55rem; border-radius: 999px;
              background:#EEF2FF; color:#3730A3; font-weight:600; font-size:.8rem; }
      .logo-wrap { display:flex; align-items:center; gap:.6rem; }
      .logo-wrap img { border-radius: 4px; }
      section[data-testid="stSidebar"] { width: 340px !important; }
      .sidebar-caption { margin: .3rem 0 .4rem 0; color:#6b7280; font-size:.92rem; }

      /* --- RHS "sidebar" panel look/feel --- */
      .rhs-aside {
        position: sticky; top: 10px;
        max-height: calc(100vh - 20px);
        overflow: auto;
        border-left: 1px solid #E5E7EB;
        padding-left: 8px;
      }
      .rhs-aside .stMarkdown, .rhs-aside .stForm, .rhs-aside .stButton, .rhs-aside .stTextInput,
      .rhs-aside .stTextArea, .rhs-aside .stNumberInput, .rhs-aside .stSelectbox { font-size: 0.95rem; }
      .rhs-aside .card { border-color:#E5E7EB; }
      .rhs-header { display:flex; align-items:center; justify-content:space-between; margin:.25rem 0 .5rem; }

      .card-title-row{display:flex;align-items:center;justify-content:space-between;margin:0 0 .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-tag">{APP_TAGLINE}</div>', unsafe_allow_html=True)
st.write("")

# ---------------------- Storage, schema & utils (New) ----------------------
SCHEMA_VERSION = 1          # bump when formats change
BACKUP_COUNT   = 3          # keep last N backups

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if path.exists():
        path.replace(bak)
        # prune backups
        baks = sorted(path.parent.glob(path.name + ".bak.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in baks[BACKUP_COUNT:]:
            try: old.unlink()
            except Exception: pass
    tmp.replace(path)

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _migrate_connections_store(raw: dict) -> dict:
    """
    Accepts legacy {conn_id: {profile: {k:v}}} and:
      - wraps into {"_schema":..., "profiles": {...}}
      - converts "env:VAR" -> {"$env":"VAR"} for secrets
    """
    if not raw:
        return {"_schema": {"version": SCHEMA_VERSION}, "profiles": {}}

    # already modern
    if "profiles" in raw and "_schema" in raw:
        # normalize any lingering "env:" strings
        def fix_env(v):
            if isinstance(v, str) and v.startswith("env:"):
                return {"$env": v.split(":",1)[1].strip()}
            return v
        out = deepcopy(raw)
        for cid, profs in (out.get("profiles") or {}).items():
            for pname, entry in (profs or {}).items():
                cfg = entry.get("config") or {}
                for k, v in list(cfg.items()):
                    cfg[k] = fix_env(v)
                entry["config"] = cfg
                entry["_meta"] = entry.get("_meta") or {"created_at": _now_iso(), "updated_at": _now_iso(), "last_test": None}
        out["_schema"]["version"] = SCHEMA_VERSION
        return out

    # legacy -> modern
    new = {"_schema": {"version": SCHEMA_VERSION}, "profiles": {}}
    for cid, profs in (raw or {}).items():
        new["profiles"].setdefault(cid, {})
        for pname, cfg in (profs or {}).items():
            cfg2 = {}
            for k, v in (cfg or {}).items():
                if isinstance(v, str) and v.startswith("env:"):
                    cfg2[k] = {"$env": v.split(":",1)[1].strip()}
                else:
                    cfg2[k] = v
            new["profiles"][cid][pname] = {
                "_meta": {
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "last_test": None  # {"ok": bool, "msg": str, "at": iso}
                },
                "config": cfg2
            }
    return new

def _connections_view_from_raw(raw: dict) -> Dict[str, Dict[str, Any]]:
    """Return legacy view {conn_id: {profile: config}} for existing callers."""
    view: Dict[str, Dict[str, Any]] = {}
    for cid, profs in (raw.get("profiles") or {}).items():
        view[cid] = {pname: (entry.get("config") or {}) for pname, entry in (profs or {}).items()}
    return view

def _load_all() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Back-compat view loader; caches raw in session."""
    raw = _read_json(CONN_STORE)
    migrated = _migrate_connections_store(raw)
    st.session_state["_CONN_STATE"] = migrated
    return _connections_view_from_raw(migrated)

def _save_all(data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Accept legacy view and write modern schema with metadata."""
    raw = st.session_state.get("_CONN_STATE") or {"_schema": {"version": SCHEMA_VERSION}, "profiles": {}}
    raw["profiles"] = raw.get("profiles") or {}
    for cid, profs in (data or {}).items():
        raw["profiles"].setdefault(cid, {})
        for pname, cfg in (profs or {}).items():
            entry = raw["profiles"][cid].get(pname) or {"_meta": {"created_at": _now_iso(), "last_test": None}}
            meta = entry.setdefault("_meta", {})
            meta["updated_at"] = _now_iso()
            entry["config"] = cfg
            raw["profiles"][cid][pname] = entry
    raw["_schema"]["version"] = SCHEMA_VERSION
    _write_json_atomic(CONN_STORE, raw)
    st.session_state["_CONN_STATE"] = raw  # keep in sync

def _resolve_secrets(cfg: Dict[str, Any], secret_keys: List[str]) -> Dict[str, Any]:
    """Turn {'$env':'NAME'} into os.getenv('NAME')."""
    out = {}
    for k, v in cfg.items():
        if k in secret_keys and isinstance(v, dict) and "$env" in v:
            out[k] = os.getenv(v["$env"], "")
        else:
            out[k] = v
    return out

# ---------------------- Utilities (masked preview, logos, etc.) ----------------------
def _mask(val: Any) -> Any:
    if not isinstance(val, str) or not val:
        return val
    if len(val) <= 6:
        return "•" * len(val)
    return f"{val[:2]}{'•' * (len(val)-4)}{val[-2:]}"

def _logo_html(basename: str, size: int = 22) -> Optional[str]:
    if not ASSETS_DIR.exists():
        return None
    base = basename.lower().replace(" ", "").replace("/", "").replace("-", "")
    for ext in (".svg", ".png", ".jpg", ".jpeg", ".webp"):
        p = ASSETS_DIR / f"{base}{ext}"
        if p.exists():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                mime = "svg+xml" if ext == ".svg" else ext[1:]
                return f'<img src="data:image/{mime};base64,{b64}" width="{size}" height="{size}" />'
            except Exception:
                return None
    return None

def _env_snippet(conn_id: str, profile: str, cfg: Dict[str, Any], secret_keys: List[str]) -> str:
    """Show export lines; secrets are masked unless they are ENV references."""
    lines = []
    prefix = f"{conn_id}_{profile}".upper().replace("-", "_").replace(" ", "_")
    for k, v in cfg.items():
        key = f"{prefix}_{k.upper()}"
        shown = v
        if k in secret_keys:
            if isinstance(v, dict) and "$env" in v:
                shown = f"${{{v['$env']}}}"  # reference
            else:
                shown = _mask(v)
        lines.append(f'export {key}="{shown}"')
    return "\n".join(lines)

def _dsn_preview(conn_id: str, cfg: Dict[str, Any]) -> str:
    try:
        if conn_id == "postgres":
            return f"postgresql://{cfg.get('user')}:***@{cfg.get('host')}:{cfg.get('port',5432)}/{cfg.get('database')}"
        if conn_id == "mysql":
            return f"mysql://{cfg.get('user')}:***@{cfg.get('host')}:{cfg.get('port',3306)}/{cfg.get('database')}"
        if conn_id == "mssql":
            drv = cfg.get('driver','ODBC Driver 18 for SQL Server')
            return f"mssql+pyodbc://{cfg.get('user')}:***@{cfg.get('server')}/{cfg.get('database')}?driver={drv}"
        if conn_id == "oracle":
            return f"oracle+oracledb://{cfg.get('user')}:***@{cfg.get('dsn')}"
        if conn_id == "sqlite":
            return f"sqlite:///{cfg.get('filepath')}"
        if conn_id == "trino":
            return f"trino://{cfg.get('user')}@{cfg.get('host')}:{cfg.get('port')}/{cfg.get('catalog')}/{cfg.get('schema')}"
        if conn_id == "duckdb":
            return f"duckdb:///{cfg.get('filepath')}"
        if conn_id == "snowflake":
            return f"snowflake://{cfg.get('user')}:***@{cfg.get('account')}/{cfg.get('database')}/{cfg.get('schema')}?warehouse={cfg.get('warehouse')}&role={cfg.get('role')}"
        if conn_id == "bigquery":
            return f"bigquery://{cfg.get('project_id')} (Service Account JSON provided)"
        if conn_id == "redshift":
            return f"redshift+psycopg2://{cfg.get('user')}:***@{cfg.get('host')}:{cfg.get('port',5439)}/{cfg.get('database')}"
        if conn_id == "synapse":
            return f"mssql+pyodbc://{cfg.get('user')}:***@{cfg.get('server')}/{cfg.get('database')}?driver=ODBC+Driver+18+for+SQL+Server"
        if conn_id == "mongodb":
            return f"{cfg.get('uri')}"
        if conn_id == "cassandra":
            return f"cassandra://{cfg.get('contact_points')}:{cfg.get('port')}/{cfg.get('keyspace')}"
        if conn_id == "redis":
            return f"redis://{cfg.get('host')}:{cfg.get('port')}/{cfg.get('db','0')}"
        if conn_id == "dynamodb":
            return f"dynamodb://{cfg.get('region_name') or 'region'} (keys provided)"
        if conn_id == "neo4j":
            return f"{cfg.get('uri')} (neo4j user provided)"
        if conn_id == "elasticsearch":
            return f"elasticsearch://{cfg.get('hosts')}"
        if conn_id == "cosmos":
            return f"cosmos://{cfg.get('endpoint')}"
        if conn_id == "firestore":
            return f"firestore://{cfg.get('project_id')} (Service Account JSON provided)"
        if conn_id == "bigtable":
            return f"bigtable://{cfg.get('project_id')}/{cfg.get('instance_id')} (SA JSON provided)"
        if conn_id == "s3":
            return f"s3://{cfg.get('bucket') or ''} ({'region ' + cfg.get('region_name') if cfg.get('region_name') else 'region not set'})"
        if conn_id == "azureblob":
            return f"azblob://{cfg.get('account_name') or 'account'}"
        if conn_id == "adls":
            return f"abfs://{cfg.get('filesystem') or ''}@{cfg.get('account_name') or 'account'}.dfs.core.windows.net"
        if conn_id == "gcs":
            return f"gcs://{cfg.get('bucket') or ''} (project {cfg.get('project_id')})"
        if conn_id == "hdfs":
            return f"webhdfs://{cfg.get('host')}:{cfg.get('port')}"
        if conn_id == "kafka":
            return f"kafka://{cfg.get('bootstrap_servers')} (security={cfg.get('security_protocol')})"
        if conn_id == "rabbitmq":
            return f"{cfg.get('amqp_url')}"
        if conn_id == "eventhubs":
            return f"eventhubs://{cfg.get('eventhub') or ''}"
        if conn_id == "pubsub":
            return f"pubsub://{cfg.get('project_id')} (SA JSON provided)"
        if conn_id == "kinesis":
            return f"kinesis://{cfg.get('region_name') or ''} (keys provided)"
        if conn_id == "spark":
            return f"spark://{cfg.get('master')} (app_name={cfg.get('app_name')})"
        if conn_id == "dask":
            return f"dask://{cfg.get('scheduler_address')}"
        if conn_id == "salesforce":
            return f"salesforce://{cfg.get('domain') or 'login' } (username provided)"
        if conn_id == "servicenow":
            return f"servicenow://{cfg.get('instance')}"
        if conn_id == "jira":
            return f"jira://{cfg.get('server')}"
        if conn_id == "sharepoint":
            return f"msgraph://tenant={cfg.get('tenant_id')}"
        if conn_id == "tableau":
            return f"tableau://{cfg.get('server')}/{cfg.get('site_id') or ''}"
        if conn_id == "gmail":
            return "gmail:// (OAuth/Service Account JSON provided)"
        if conn_id == "msgraph":
            return f"msgraph://tenant={cfg.get('tenant_id')}"
        if conn_id == "weaviate":
            cluster_url = (cfg.get("cluster_url") or cfg.get("url") or "").strip().rstrip("/")
            if cluster_url:
                http_url = cluster_url
            else:
                scheme = (cfg.get("scheme") or "https").lower()
                host = (cfg.get("host") or "").strip()
                port = int(cfg.get("port") or (443 if scheme == "https" else 80))
                default_port = 443 if scheme == "https" else 80
                http_url = f"{scheme}://{host}" + ("" if port == default_port else f":{port}")
            mt = " (multi-tenancy)" if str(cfg.get("multi_tenancy") or "").lower() in ("true", "yes", "1") else ""
            return f"weaviate://{http_url}{mt}"
    except Exception:
        pass
    return "(preview unavailable)"

def _short_timeout_env():
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

def _config_signature(cfg: Dict[str, Any]) -> str:
    try:
        return json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(cfg)

def _get_state_keys(conn_id: str) -> Dict[str, str]:
    return {"ok": f"{conn_id}_last_test_ok", "msg": f"{conn_id}_last_test_msg", "sig": f"{conn_id}_last_test_sig"}

def _status_cache_key(conn_id: str, profile: str) -> str:
    return f"status::{conn_id}::{profile}"

def _cache_status_set(conn_id: str, profile: str, ok: Optional[bool], msg: str) -> None:
    st.session_state[_status_cache_key(conn_id, profile)] = {"ok": ok, "msg": msg}

def _cache_status_get(conn_id: str, profile: str) -> Dict[str, Any]:
    return st.session_state.get(_status_cache_key(conn_id, profile), {"ok": None, "msg": "Not tested"})

# ---------------------- Schema / Registry ----------------------
@dataclass
class Field:
    key: str
    label: str
    required: bool = False
    kind: str = "text"  # text | password | int | textarea | select
    placeholder: str = ""
    options: Optional[List[str]] = None

@dataclass
class Connector:
    id: str
    name: str
    icon: str
    fields: List[Field]
    secret_keys: List[str]
    category: str
    logo_key: Optional[str] = None

def F(key: str, label: str, **kw) -> Field:
    return Field(key=key, label=label, **kw)

# ---------------------- Registry (unchanged list) ----------------------
REGISTRY: List[Connector] = [
    # ... (registry identical to your current file; omitted here for brevity)
]

# ---- SAFETY FALLBACK: ensure registry is not empty so the app can boot
if not REGISTRY:
    REGISTRY = [
        Connector(
            id="sqlite",
            name="SQLite",
            icon="🗂️",
            fields=[Field("filepath", "DB File Path", required=True, placeholder="./my.db")],
            secret_keys=[],
            category="SQL",
            logo_key="sqlite",
        )
    ]

REG_BY_ID: Dict[str, Connector] = {c.id: c for c in REGISTRY}

# ---------------------- Sidebar (SPA navigation) ----------------------
def _sorted_filtered_connectors(q: str) -> List[Connector]:
    items = sorted(REGISTRY, key=lambda x: x.name.lower())
    if not q:
        return items
    ql = q.lower()
    return [c for c in items if ql in c.name.lower() or ql in c.id.lower() or ql in c.category.lower()]

# Session state defaults (guarded)
default_id = REGISTRY[0].id if REGISTRY else None
if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = default_id
st.session_state.setdefault("rhs_open", False)

def _set_active(conn_id: str):
    st.session_state["selected_id"] = conn_id
    st.session_state["rhs_open"] = True

with st.sidebar:
    logo_path = ASSETS_DIR / "logo.png"
    if logo_path.is_file():
        st.image(str(logo_path), caption="iSOFT ANZ Pvt Ltd", use_container_width=True)
    else:
        st.caption("")

    st.markdown("### 🔎 Search")
    q = st.text_input("Search connectors", placeholder="snowflake, postgres, blob, kafka...").strip()
    st.markdown('<div class="sidebar-caption">All connectors</div>', unsafe_allow_html=True)

    filtered = _sorted_filtered_connectors(q)
    if not filtered:
        st.caption("No connectors registered (using fallback).")
    for c in filtered:
        is_active = (st.session_state["selected_id"] == c.id and st.session_state["rhs_open"])
        if st.button(f"{c.icon}  {c.name}", key=f"nav_{c.id}",
                     type=("primary" if is_active else "secondary"),
                     use_container_width=True):
            _set_active(c.id)

    # ---------- Import / Export Connections ----------
    st.divider()
    with st.expander("🔄 Import / Export Connections", expanded=False):
        _ = _load_all()  # ensure _CONN_STATE up to date
        raw = st.session_state.get("_CONN_STATE") or {"_schema": {"version": SCHEMA_VERSION}, "profiles": {}}

        # Export (modern with metadata)
        st.markdown("**Export connections**")
        export_json = json.dumps(raw, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download connections.json",
            data=export_json,
            file_name="connections.json",
            mime="application/json",
            use_container_width=True,
        )

        st.write("")
        # Import
        st.markdown("**Import connections**")
        up = st.file_uploader("Upload a connections.json", type=["json"])
        if up:
            try:
                data = json.loads(up.read().decode("utf-8"))
                # accept either modern {"_schema","profiles"} or legacy {cid:{p:{}}}
                if "profiles" in data and "_schema" in data:
                    merged = st.session_state.get("_CONN_STATE") or {"_schema": {"version": SCHEMA_VERSION}, "profiles": {}}
                    # shallow merge profiles
                    for cid, profs in (data.get("profiles") or {}).items():
                        merged["profiles"].setdefault(cid, {})
                        merged["profiles"][cid].update(profs or {})
                    merged["_schema"]["version"] = SCHEMA_VERSION
                    _write_json_atomic(CONN_STORE, merged)
                    st.session_state["_CONN_STATE"] = merged
                    st.success("Imported connections successfully.")
                    st.rerun()
                elif isinstance(data, dict):
                    # legacy import -> write modern via _save_all
                    merged_view = _load_all()
                    for cid, profs in (data or {}).items():
                        merged_view.setdefault(cid, {})
                        merged_view[cid].update(profs or {})
                    _save_all(merged_view)
                    st.success("Imported connections (legacy) successfully.")
                    st.rerun()
                else:
                    st.error("Invalid format.")
            except Exception as e:
                st.error(f"Failed to import: {e}")

    # ---------- Import / Export Pipelines ----------
    with st.expander("🔄 Import / Export Pipelines", expanded=False):
        existing = None  # allocate after defining pipeline helpers

# Resolve current connector from state (guarded)
active_id = st.session_state.get("selected_id")
conn: Optional[Connector] = REG_BY_ID.get(active_id) if active_id else None

# ---------------------- Load store & KPIs ----------------------
all_profiles = _load_all()
total_profiles_all = sum(len(v) for v in all_profiles.values())

k2, k3 = st.columns([1,1])
with k2:
    st.markdown(f'<div class="kpi">🧩 Connectors: <b>{len(REGISTRY)}</b></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi">🗂️ Profiles: <b>{total_profiles_all}</b></div>', unsafe_allow_html=True)

st.write("")
main_left, main_right = st.columns([7, 5], gap="large")

def _prefill_and_open_editor(conn_id: str, profile: str, cfg: Dict[str, Any]) -> None:
    """Open the RHS editor with fields prefilled from an existing saved profile."""
    st.session_state["selected_id"] = conn_id
    st.session_state["rhs_open"] = True
    st.session_state[f"{conn_id}_profile_name"] = profile
    for f in (REG_BY_ID.get(conn_id).fields if REG_BY_ID.get(conn_id) else []):
        st.session_state[f"{conn_id}_{f.key}"] = cfg.get(f.key, "")

# ---------------------- Header renderer ----------------------
def render_header(container, conn: Connector, total_connectors: int, total_profiles: int, *, in_rhs: bool):
    with container:
        thumb = _logo_html(conn.logo_key or conn.id, size=22)
        if thumb:
            container.markdown(
                f'<div class="logo-wrap">{thumb}<h2 style="margin:0;">{conn.icon} {conn.name}</h2></div>',
                unsafe_allow_html=True,
            )
        else:
            container.markdown(f"## {conn.icon} {conn.name}")
        container.caption(f"Category: **{conn.category}** · ID: `{conn.id}`")
        if in_rhs:
            container.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:0.4rem 0 0.6rem;'>",
                               unsafe_allow_html=True)

# ---------------------- Test handlers (unchanged) ----------------------
# Keep all your test_* functions and TEST_HANDLERS dict exactly as in your file.
# (No changes needed to handler bodies.)

# ---------------------- Configure form (with ENV-secrets UX) ----------------------
def render_configure_form(container, conn: Connector):
    with container:
        title_col, close_col = st.columns([10, 1])
        with title_col:
            st.markdown(
                """
                <style>
                  .rowhead { font-size:.9rem; color:#6b7280; margin-bottom:.25rem; }
                  .action-col button[kind="secondary"] { padding: .3rem .5rem; }
                </style>
                <div class="card all-configured"><h3>Configure connection profile</h3>
                """,
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button("✖", key="close_rhs", type="secondary", help="Close", use_container_width=True):
                st.session_state["rhs_open"] = False
                st.rerun()

        _keys = _get_state_keys(conn.id)
        for k in _keys.values():
            st.session_state.setdefault(k, None)

        with st.form(key=f"form_{conn.id}", clear_on_submit=False):
            profile_name = st.text_input("Profile Name", placeholder="dev / staging / prod", key=f"{conn.id}_profile_name")

            values: Dict[str, Any] = {}
            missing_required: List[str] = []
            for f in conn.fields:
                ikey = f"{conn.id}_{f.key}"
                if f.kind == "password" or f.key in conn.secret_keys:
                    col_pwd, col_env = st.columns([3,2])
                    with col_pwd:
                        raw_val = st.text_input(f.label, type="password", placeholder=f.placeholder or "", key=ikey)
                    with col_env:
                        use_env = st.checkbox("Use ENV", key=ikey+"_useenv", help="Reference an environment variable instead of storing a secret")
                        env_name = ""
                        if use_env:
                            env_name = st.text_input("ENV var", placeholder="MY_SECRET_VAR", key=ikey+"_envname")
                    if st.session_state.get(ikey+"_useenv") and (env_name or "").strip():
                        val = {"$env": env_name.strip()}
                    else:
                        val = raw_val
                elif f.kind == "int":
                    val = st.number_input(f.label, min_value=0, step=1, value=0, key=ikey)
                elif f.kind == "textarea":
                    val = st.text_area(f.label, placeholder=f.placeholder or "", height=120, key=ikey)
                elif f.kind == "select":
                    opts = f.options or [""]
                    val = st.selectbox(f.label, opts, index=0, key=ikey)
                else:
                    val = st.text_input(f.label, placeholder=f.placeholder or "", key=ikey)

                values[f.key] = val
                if f.required and (val is None or (isinstance(val, str) and val.strip() == "")):
                    missing_required.append(f.label)

            c1, c2, c3, c4 = st.columns([1,1,1,1])
            submitted    = c1.form_submit_button("💾 Save Profile", use_container_width=True)
            preview      = c2.form_submit_button("🧪 Preview DSN", use_container_width=True)
            envvars      = c3.form_submit_button("🔐 Env-Vars Snippet", use_container_width=True)
            test_clicked = c4.form_submit_button("✅ Test Connection", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if test_clicked:
            handler = TEST_HANDLERS.get(conn.id)
            if not handler:
                st.warning("Test handler not implemented for this connector.")
                st.session_state[_keys["ok"]] = False
                st.session_state[_keys["msg"]] = "No test available."
                st.session_state[_keys["sig"]] = _config_signature(values)
            else:
                _short_timeout_env()
                with st.spinner("Testing connection..."):
                    resolved = _resolve_secrets(values, REG_BY_ID[conn.id].secret_keys if REG_BY_ID.get(conn.id) else [])
                    ok, msg = handler(resolved)
                st.session_state[_keys["ok"]] = bool(ok)
                st.session_state[_keys["msg"]] = msg
                st.session_state[_keys["sig"]] = _config_signature(values)

                # persist last_test to store metadata if profile name present
                store_raw = st.session_state.get("_CONN_STATE")
                profname = (st.session_state.get(f"{conn.id}_profile_name") or "").strip()
                if store_raw and profname:
                    store_raw.setdefault("profiles", {}).setdefault(conn.id, {})
                    entry = store_raw["profiles"][conn.id].setdefault(profname, {"_meta": {"created_at": _now_iso(), "last_test": None}})
                    meta = entry.setdefault("_meta", {})
                    meta["updated_at"] = _now_iso()
                    meta["last_test"] = {"ok": bool(ok), "msg": str(msg), "at": _now_iso()}
                    entry["config"] = values
                    _write_json_atomic(CONN_STORE, store_raw)
                    st.session_state["_CONN_STATE"] = store_raw

        last_ok  = st.session_state.get(_keys["ok"])
        last_msg = st.session_state.get(_keys["msg"]) or "Not tested"
        sig_now  = _config_signature(values)
        sig_last = st.session_state.get(_keys["sig"])

        tested_at = None
        store_raw = st.session_state.get("_CONN_STATE") or {}
        profname = (st.session_state.get(f"{conn.id}_profile_name") or "").strip()
        try:
            tested_at = (((store_raw.get("profiles") or {}).get(conn.id) or {}).get(profname) or {}).get("_meta", {}).get("last_test", {}).get("at")
        except Exception:
            tested_at = None

        if last_ok is True and sig_now == sig_last:
            when = f' <span class="muted small" title="Tested at {tested_at} UTC">({tested_at or "just now"})</span>' if tested_at else ""
            st.markdown(f'<span class="pill">✅ Successful</span> <span class="muted small">{last_msg}</span>{when}', unsafe_allow_html=True)
        elif last_ok is False and sig_now == sig_last:
            st.markdown(f'<span class="pill" style="background:#FEE2E2;color:#991B1B;">❌ Failed</span> <span class="muted small">{last_msg}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill" style="background:#FFF7ED;color:#9A3412;">⏳ Not tested</span> <span class="muted small">Run “Test Connection” before saving.</span>', unsafe_allow_html=True)

        if preview:
            st.info("Indicative DSN/URI preview (no network calls):")
            st.code(_dsn_preview(conn.id, _resolve_secrets(values, REG_BY_ID[conn.id].secret_keys if REG_BY_ID.get(conn.id) else [])), language="text")

        if envvars:
            st.info("Copy/paste into your shell (masked preview; ENV refs remain dynamic):")
            st.code(_env_snippet(conn.id, profile_name or "PROFILE", values, REG_BY_ID[conn.id].secret_keys if REG_BY_ID.get(conn.id) else []), language="bash")

        if submitted:
            if not profile_name.strip():
                st.error("Please provide a **Profile Name**.")
            elif missing_required:
                st.error("Missing required fields: " + ", ".join(missing_required))
            elif not (st.session_state.get(_keys["ok"]) is True and st.session_state.get(_keys["sig"]) == _config_signature(values)):
                st.error("Please **Test Connection** and ensure it is **Successful** for the current values before saving.")
            else:
                all_profiles = _load_all()
                all_profiles.setdefault(conn.id, {})
                all_profiles[conn.id][profile_name] = values
                _save_all(all_profiles)
                _cache_status_set(conn.id, profile_name, st.session_state.get(_keys["ok"]), st.session_state.get(_keys["msg"]) or "")
                st.success(f"Saved **{profile_name}** for {conn.icon} {conn.name}.")

# ---------------------- RHS panel ----------------------
if st.session_state["rhs_open"] and conn is not None:
    with main_right:
        st.markdown('<div class="rhs-aside">', unsafe_allow_html=True)
        render_header(main_right, conn, len(REGISTRY), total_profiles_all, in_rhs=True)
        render_configure_form(main_right, conn)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with main_right:
        st.markdown("")

# ---------------------- All Configured Connections (left) ----------------------
def _run_status_check_for_all():
    _short_timeout_env()
    for cid, items in (_load_all() or {}).items():
        handler = TEST_HANDLERS.get(cid)
        for pname, cfg in (items or {}).items():
            if handler:
                try:
                    secret_keys = REG_BY_ID.get(cid).secret_keys if REG_BY_ID.get(cid) else []
                    resolved = _resolve_secrets(cfg, secret_keys)
                    ok, msg = handler(resolved)
                except Exception as e:
                    ok, msg = False, str(e)
            else:
                ok, msg = None, "No test implemented."
            _cache_status_set(cid, pname, ok, msg)

with main_left:
    st.markdown(
        """
        <style>
          .rowhead { font-size:.9rem; color:#6b7280; margin-bottom:.25rem; }
          .action-col button[kind="secondary"] { padding: .3rem .5rem; }
        </style>
        <div class="card all-configured"><h3>📚 All configured connections</h3>
        """,
        unsafe_allow_html=True,
    )

    _run_status_check_for_all()
    all_profiles = _load_all()

    if not all_profiles:
        st.info("You haven’t saved any connections yet.")
    else:
        h1, h2, h3, h4, h5 = st.columns([4, 3, 3, 1, 1])
        h1.markdown('<div class="rowhead">Connector</div>', unsafe_allow_html=True)
        h2.markdown('<div class="rowhead">Profile</div>', unsafe_allow_html=True)
        h3.markdown('<div class="rowhead">Status</div>', unsafe_allow_html=True)
        h4.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)
        h5.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)

        for cid in sorted(all_profiles.keys(), key=lambda c: (REG_BY_ID.get(c).name if REG_BY_ID.get(c) else c).lower()):
            meta = REG_BY_ID.get(cid)
            for pname in sorted(all_profiles[cid].keys(), key=lambda x: x.lower()):
                cfg = all_profiles[cid][pname]
                status_info = _cache_status_get(cid, pname)
                ok = status_info.get("ok")
                status_text = (
                    '<span class="pill">✅ Successful</span>' if ok is True
                    else '<span class="pill" style="background:#FEE2E2;color:#991B1B;">❌ Failed</span>' if ok is False
                    else '<span class="pill" style="background:#FFF7ED;color:#9A3412;">⏳ Not tested</span>'
                )

                c1, c2, c3, c4, c5 = st.columns([4, 3, 3, 1, 1])
                c1.markdown(f"{meta.icon if meta else '🔌'} **{meta.name if meta else cid}**")
                c2.markdown(f"`{pname}`")
                c3.markdown(status_text, unsafe_allow_html=True)

                # EDIT
                if c4.button("📝", key=f"edit::{cid}::{pname}", help="Edit this profile"):
                    _prefill_and_open_editor(cid, pname, cfg)
                    st.rerun()

                # DELETE (guard if pipelines use this profile)
                if c5.button("🗑️", key=f"del::{cid}::{pname}", help="Delete this profile"):
                    # guard: used by any pipeline?
                    pipes = None
                    try:
                        pipes = _read_json(PIPE_STORE)
                        if "_schema" in pipes and "items" in pipes:
                            pipes = pipes["items"]
                    except Exception:
                        pipes = None
                    in_use = []
                    for _pid, p in (pipes or {}).items():
                        if (p.get("source_connector")==cid and p.get("source_profile")==pname) or \
                           (cid=="weaviate" and p.get("destination_profile")==pname):
                            in_use.append(p.get("name") or _pid)
                    if in_use:
                        st.error("Cannot delete. This profile is used by pipelines: " + ", ".join(in_use))
                    else:
                        store = _load_all()
                        try:
                            if cid in store and pname in store[cid]:
                                del store[cid][pname]
                                if not store[cid]:
                                    del store[cid]
                                _save_all(store)
                                st.success(f"Deleted profile **{pname}** for {meta.icon if meta else '🔌'} {meta.name if meta else cid}.")
                                st.rerun()
                            else:
                                st.warning("Profile not found (already deleted?).")
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================================================
# Pipelines helpers & UI (drop into your Streamlit page)
# =================================================================================================

# ---------------------------------------------------------------------------------
# Connections store helpers (read-only view here)
# ---------------------------------------------------------------------------------
def _load_profiles_store() -> dict:
    """Return legacy view for pipelines: {cid: {profile: cfg}}"""
    raw = _read_json(CONN_STORE)
    migrated = _migrate_connections_store(raw)
    return _connections_view_from_raw(migrated)

def _profiles_for_connector(conn_id: str) -> List[str]:
    store = _load_profiles_store()
    return sorted(list((store.get(conn_id) or {}).keys()), key=str.lower)

def _profiles_for(connector_id: str) -> List[str]:
    store = _load_profiles_store()
    profs = store.get(connector_id, {}) or {}
    return sorted(profs.keys(), key=str.lower)

# ---------------------------------------------------------------------------------
# Pipelines store helpers (atomic + schema)
# ---------------------------------------------------------------------------------
def _pipelines_load_all() -> Dict[str, Dict[str, Any]]:
    raw = _read_json(PIPE_STORE) or {}
    if "_schema" not in raw:
        raw = {"_schema": {"version": SCHEMA_VERSION}, "items": raw or {}}
    raw["_schema"]["version"] = SCHEMA_VERSION
    st.session_state["_PIPE_STATE"] = raw
    return raw.get("items") or {}

def _pipelines_save_all(data: Dict[str, Dict[str, Any]]) -> None:
    raw = st.session_state.get("_PIPE_STATE") or {"_schema": {"version": SCHEMA_VERSION}, "items": {}}
    raw["items"] = data or {}
    raw["_schema"]["version"] = SCHEMA_VERSION
    _write_json_atomic(PIPE_STORE, raw)
    st.session_state["_PIPE_STATE"] = raw

def _pipeline_defaults() -> Dict[str, Any]:
    return {
        "name": "",
        "source_connector": "",
        "source_profile": "",
        "destination_connector": "weaviate",
        "destination_profile": "",
        "collection": "Documents",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_docs": 50,
        "notes": "",
    }

# ---------------------------------------------------------------------------------
# Session defaults for pipeline editor
# ---------------------------------------------------------------------------------
st.session_state.setdefault("editing_pipeline_id", None)
st.session_state.setdefault("pipeline_form_open", False)

# =================================================================================================
# UI: Pipelines Card
# =================================================================================================
with st.container(border=True):
    st.markdown("### 🧪 Pipelines")

    c_a, c_b = st.columns([5, 1])
    with c_a:
        st.caption("Create, save, edit, delete and manually run pipelines that move data from a source connector to Weaviate.")
    with c_b:
        if st.button("➕ New Pipeline", use_container_width=True, key="btn_new_pipeline"):
            st.session_state["editing_pipeline_id"] = None
            st.session_state["pipeline_form_open"] = True

    pipelines = _pipelines_load_all()

    if st.session_state["pipeline_form_open"]:
        is_edit = st.session_state["editing_pipeline_id"] is not None
        pid = st.session_state["editing_pipeline_id"]
        initial = pipelines.get(pid, _pipeline_defaults())
        st.subheader("Edit Pipeline" if is_edit else "Create Pipeline")

        # -------------------- SOURCE CARD --------------------
        st.session_state.setdefault("pipe_src_connector", initial.get("source_connector", ""))
        st.session_state.setdefault("pipe_src_profile",   initial.get("source_profile", ""))
        st.session_state.setdefault("pipe__last_src_connector", st.session_state["pipe_src_connector"])

        with st.container(border=True):
            st.markdown("**Source**")

            profiles_store = _load_profiles_store()
            avail_sources = [cid for cid, profs in profiles_store.items() if profs]
            avail_sources = [c for c in avail_sources if c != "weaviate"] or avail_sources

            src_connector = st.selectbox(
                "Source Connector",
                options=[""] + avail_sources,
                key="pipe_src_connector",
            )

            if st.session_state.get("pipe__last_src_connector") != src_connector:
                st.session_state["pipe__last_src_connector"] = src_connector
                st.session_state["pipe_src_profile"] = ""
                st.rerun()

            src_profiles = _profiles_for_connector(src_connector) if src_connector else []
            st.selectbox(
                "Source Profile",
                options=[""] + src_profiles if src_profiles else [""],
                key="pipe_src_profile",
            )

            if src_connector and not src_profiles:
                st.caption("No saved profiles for this connector. Open the connector, test, then save a profile.")
                jump_col1, _ = st.columns([1, 3])
                if jump_col1.button("Create profile now"):
                    st.session_state["selected_id"] = src_connector
                    st.session_state["rhs_open"] = True
                    st.rerun()

        # -------------------- FORM (NAME, DESTINATION, PROCESSING, SAVE) --------------------
        with st.form("pipeline_form", clear_on_submit=False):
            name = st.text_input("Pipeline Name", value=initial.get("name", ""), placeholder="My First Pipeline")

            with st.container(border=True):
                st.markdown("**Destination (Weaviate)**")
                dest_connector = "weaviate"
                weaviate_profiles = _profiles_for_connector(dest_connector)
                dest_profile = st.selectbox(
                    "Weaviate Profile",
                    options=([""] + weaviate_profiles) if weaviate_profiles else [""],
                    index=([""] + weaviate_profiles).index(initial.get("destination_profile", ""))
                          if weaviate_profiles and initial.get("destination_profile", "") in ([""] + weaviate_profiles)
                          else 0,
                )
                collection = st.text_input(
                    "Weaviate Collection (Class) Name",
                    value=initial.get("collection", "Documents"),
                    placeholder="Documents",
                    help="Letters, digits, underscore; start with a letter/underscore.",
                )

            with st.container(border=True):
                st.markdown("**Processing Settings**")
                c1, c2 = st.columns(2)
                with c1:
                    chunk_size = st.number_input("Chunk Size (chars)", min_value=100, step=50, value=int(initial.get("chunk_size", 1000)))
                    max_docs = st.number_input("Max Docs (for quick runs)", min_value=1, step=10, value=int(initial.get("max_docs", 50)))
                with c2:
                    chunk_overlap = st.number_input("Chunk Overlap (chars)", min_value=0, step=10, value=int(initial.get("chunk_overlap", 150)))
                    embedding_model = st.text_input("Embedding Model (placeholder)", value=initial.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
                notes = st.text_area("Notes (optional)", value=initial.get("notes", ""))

            csave, ccancel = st.columns([1, 1])
            submit = csave.form_submit_button("💾 Save Pipeline", use_container_width=True)
            cancel = ccancel.form_submit_button("Cancel", use_container_width=True)

        if cancel:
            st.session_state["pipeline_form_open"] = False
            st.session_state["editing_pipeline_id"] = None
            st.rerun()

        if submit:
            src_connector = st.session_state.get("pipe_src_connector", "")
            src_profile   = st.session_state.get("pipe_src_profile", "")

            errors = []
            if not name.strip():
                errors.append("Pipeline Name is required.")
            if not src_connector:
                errors.append("Source Connector is required.")
            if not src_profile:
                errors.append("Source Profile is required.")
            if not dest_profile:
                errors.append("Weaviate Profile is required.")
            if not collection.strip():
                errors.append("Weaviate Collection is required.")

            # validate that referenced profiles still exist
            exist_all = _load_profiles_store()
            if src_connector and (src_connector not in exist_all or src_profile not in (exist_all[src_connector] or {})):
                errors.append(f"Source profile `{src_profile}` for `{src_connector}` does not exist.")
            if "weaviate" not in exist_all or dest_profile not in (exist_all["weaviate"] or {}):
                errors.append(f"Weaviate profile `{dest_profile}` does not exist.")

            if errors:
                st.error("Please fix the following:\n" + "\n".join(f"- {e}" for e in errors))
            else:
                pipelines = _pipelines_load_all()
                if is_edit:
                    pid_new = st.session_state["editing_pipeline_id"]
                else:
                    base = name.strip().lower().replace(" ", "_") or "pipeline"
                    pid_new, i = base, 2
                    while pid_new in pipelines:
                        pid_new = f"{base}_{i}"
                        i += 1

                entry = {
                    "name": name.strip(),
                    "source_connector": src_connector,
                    "source_profile": src_profile,
                    "destination_connector": "weaviate",
                    "destination_profile": dest_profile,
                    "collection": collection.strip(),
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                    "embedding_model": embedding_model.strip(),
                    "max_docs": int(max_docs),
                    "notes": notes,
                    "_meta": {
                        "created_at": pipelines.get(pid_new, {}).get("_meta", {}).get("created_at", _now_iso()),
                        "updated_at": _now_iso(),
                        "last_run": pipelines.get(pid_new, {}).get("_meta", {}).get("last_run", None),
                    }
                }
                pipelines[pid_new] = entry
                _pipelines_save_all(pipelines)
                st.success(f"Saved pipeline **{name}**.")
                st.session_state["pipeline_form_open"] = False
                st.session_state["editing_pipeline_id"] = None
                st.rerun()

    # --- List all pipelines ---
    pipelines = _pipelines_load_all()
    if not pipelines:
        st.info("No pipelines yet. Click **New Pipeline** to create one.")
    else:
        st.markdown("**Saved Pipelines**")
        h1, h2, h3, h4, h5 = st.columns([4, 3, 2, 1, 2])
        h1.markdown('<div class="rowhead">Name</div>', unsafe_allow_html=True)
        h2.markdown('<div class="rowhead">Source → Destination</div>', unsafe_allow_html=True)
        h3.markdown('<div class="rowhead">Collection</div>', unsafe_allow_html=True)
        h4.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)
        h5.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)

        for pid in sorted(pipelines.keys(), key=lambda x: pipelines[x]["name"].lower()):
            p = pipelines[pid]
            c1, c2, c3, c4, c5 = st.columns([4, 3, 2, 1, 2])
            c1.markdown(f"**{p['name']}**")

            src_meta = REG_BY_ID.get(p["source_connector"])
            src_label = f"{(src_meta.icon + ' ' + src_meta.name) if src_meta else p['source_connector']} `{p['source_profile']}`"
            dst_label = f"🧠 Weaviate `{p['destination_profile']}`"
            c2.markdown(f"{src_label} → {dst_label}")
            c3.markdown(f"`{p['collection']}`")

            if c4.button("▶️", key=f"run::{pid}", help="Run this pipeline now"):
                exist_all = _load_profiles_store()
                ok_src = p["source_connector"] in exist_all and p["source_profile"] in (exist_all[p["source_connector"]] or {})
                ok_dst = "weaviate" in exist_all and p["destination_profile"] in (exist_all["weaviate"] or {})
                if not (ok_src and ok_dst):
                    st.error("Pipeline is misconfigured: source/destination profiles missing.")
                else:
                    st.info(f"Starting pipeline **{p['name']}**… (execution wiring will be added next)")
                    pipes = _pipelines_load_all()
                    if pid in pipes:
                        meta = pipes[pid].setdefault("_meta", {})
                        meta["last_run"] = {"status": "skipped", "msg": "executor not implemented", "at": _now_iso()}
                        _pipelines_save_all(pipes)

            e_col, d_col = c5.columns([1, 1])
            if e_col.button("📝", key=f"edit_pipe::{pid}", help="Edit"):
                st.session_state["editing_pipeline_id"] = pid
                st.session_state["pipeline_form_open"] = True
                st.rerun()

            if d_col.button("🗑️", key=f"delete_pipe::{pid}", help="Delete"):
                try:
                    allp = _pipelines_load_all()
                    if pid in allp:
                        del allp[pid]
                        _pipelines_save_all(allp)
                        st.success(f"Deleted pipeline **{p['name']}**.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Finalize Pipelines Import/Export content in sidebar expander ----
with st.sidebar:
    # Now that helpers exist
    with st.expander("🔄 Import / Export Pipelines", expanded=False):
        existing = _pipelines_load_all()

        st.markdown("**Export pipelines**")
        if existing:
            export_json = json.dumps({"_schema":{"version": SCHEMA_VERSION}, "items": existing}, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download pipelines.json",
                data=export_json, file_name="pipelines.json", mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Nothing to export yet.")

        st.write("")
        st.markdown("**Import pipelines**")
        up2 = st.file_uploader("Upload a pipelines.json", type=["json"], key="pipelines_up")
        if up2:
            try:
                data = json.loads(up2.read().decode("utf-8"))
                items = data.get("items") if isinstance(data, dict) else data
                if not isinstance(items, dict):
                    st.error("Invalid format: expected {'items': {...}}")
                else:
                    merged = _pipelines_load_all()
                    merged.update(items or {})
                    _pipelines_save_all(merged)
                    st.success("Imported pipelines successfully.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to import: {e}")
