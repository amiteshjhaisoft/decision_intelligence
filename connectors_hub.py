# Author: Amitesh Jha | iSoft | 2025-10-12
# connectors_hub.py
# Professional Connectors Hub (Streamlit)
# - Sidebar: all connectors listed alphabetically as links (icon + name)
# - Dynamic forms by connector; required-field validation
# - Persist profiles to ./connections.json (secrets masked in UI)
# - Import/Export JSON; DSN preview; Env-var snippet
# - Optional logos in ./assets (e.g., snowflake.svg, postgres.png)
# - "Test Connection" for ALL registered connectors (best-effort, short timeouts, safe)
# - NEW: RHS panel (sidebar-like) that shows the Configure form when opened via panel=profile

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from textwrap import dedent

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

      /* --- Professional sidebar link list (native links, no underline) --- */
      .sidebar-wrap{
        width:100%; background:#fff; border:1px solid #E5E7EB; border-radius:10px; overflow:hidden;
      }
      .sidebar-scroll{ max-height:520px; overflow:auto; }
      .sidebar-row{
        display:grid; grid-template-columns:26px 1fr; align-items:center;
        height:38px; padding:0 .55rem 0 .45rem;
        border-bottom:1px dashed #F1F5F9;
        text-decoration:none; border-radius:0;
        color:#111827; background:transparent;
        transition:background .12s ease-in-out, color .12s ease-in-out;
      }
      .sidebar-row:last-child{ border-bottom:0; }
      .sidebar-row:hover{ background:#F3F4F6; }
      .sidebar-row.is-active{ background:#EEF2FF; color:#3730A3; font-weight:600; }
      .sidebar-ico{ font-size:16px; opacity:.95; }
      .sidebar-name{ font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .sidebar-wrap a{ text-decoration:none !important; color:inherit !important; }

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
      .rhs-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:.3rem; }
      .rhs-close a { text-decoration:none; color:#6b7280; }
      .rhs-close a:hover { color:#111827; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="app-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-tag">{APP_TAGLINE}</div>', unsafe_allow_html=True)
st.write("")

# ---------------------- Utilities ----------------------
def _load_all() -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not CONN_STORE.exists():
        return {}
    try:
        return json.loads(CONN_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_all(data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    CONN_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _mask(val: Any) -> Any:
    if not isinstance(val, str) or not val:
        return val
    if len(val) <= 6:
        return "•" * len(val)
    return f"{val[:2]}{'•' * (len(val)-4)}{val[-2:]}"

def masked_view(d: Dict[str, Any], secret_keys: List[str]) -> Dict[str, Any]:
    return {k: (_mask(v) if k in secret_keys else v) for k, v in d.items()}

def _logo_html(basename: str, size: int = 20) -> Optional[str]:
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
    lines = []
    prefix = f"{conn_id}_{profile}".upper().replace("-", "_").replace(" ", "_")
    for k, v in cfg.items():
        key = f"{prefix}_{k.upper()}"
        shown = _mask(v) if k in secret_keys else v
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
    except Exception:
        pass
    return "(preview unavailable)"

def _short_timeout_env():
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# --- Helpers: enforce Test-before-Save & cache statuses for table ---
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

# --- Registry (unchanged) ---
REGISTRY: List[Connector] = [
    Connector("postgres", "PostgreSQL", "🐘",[F("host","Host",True), F("port","Port","int"), F("database","Database",True), F("user","User",True), F("password","Password",True,"password")],["password"],"SQL","postgres"),
    Connector("mysql", "MySQL / MariaDB", "🐬",[F("host","Host",True), F("port","Port","int"), F("database","Database",True), F("user","User",True), F("password","Password",True,"password")],["password"],"SQL","mysql"),
    Connector("mssql", "SQL Server / Azure SQL", "🪟",[F("driver","ODBC Driver",placeholder="ODBC Driver 18 for SQL Server"),F("server","Server",True),F("database","Database",True),F("user","User",True),F("password","Password",True,"password")],["password"],"SQL","mssql"),
    Connector("oracle", "Oracle", "🏺",[F("dsn","DSN",True,placeholder="host:port/service_name"),F("user","User",True),F("password","Password",True,"password")],["password"],"SQL","oracle"),
    Connector("sqlite", "SQLite", "🗂️",[F("filepath","DB File Path",True,placeholder="./my.db")],[],"SQL","sqlite"),
    Connector("trino", "Trino / Presto", "🚀",[F("host","Host",True),F("port","Port","int"),F("catalog","Catalog",True),F("schema","Schema",True),F("user","User",True)],[],"SQL","trino"),
    Connector("duckdb", "DuckDB", "🦆",[F("filepath","DB File Path",True,placeholder="./warehouse.duckdb")],[],"SQL","duckdb"),
    Connector("snowflake","Snowflake","❄️",[F("account","Account",True,placeholder="xy12345.ap-southeast-2"),F("user","User",True),F("password","Password",True,"password"),F("warehouse","Warehouse"),F("database","Database"),F("schema","Schema"),F("role","Role")],["password"],"Cloud DW / Analytics","snowflake"),
    Connector("bigquery","BigQuery","🧮",[F("project_id","Project ID",True),F("credentials_json","Service Account JSON",True,"textarea","{...}")],["credentials_json"],"Cloud DW / Analytics","bigquery"),
    Connector("redshift","Amazon Redshift","🧊",[F("host","Host",True),F("port","Port","int"),F("database","Database",True),F("user","User",True),F("password","Password",True,"password")],["password"],"Cloud DW / Analytics","redshift"),
    Connector("synapse","Azure Synapse (SQL)","🔷",[F("server","Server",True,placeholder="yourserver.database.windows.net"),F("database","Database",True),F("user","User",True),F("password","Password",True,"password")],["password"],"Cloud DW / Analytics","synapse"),
    Connector("mongodb","MongoDB","🍃",[F("uri","Mongo URI",True,placeholder="mongodb+srv://user:pass@cluster/db")],["uri"],"NoSQL / Graph / Search","mongodb"),
    Connector("cassandra","Cassandra","💠",[F("contact_points","Contact Points",True,placeholder="host1,host2"),F("port","Port","int"),F("username","Username"),F("password","Password","password"),F("keyspace","Keyspace")],["password"],"NoSQL / Graph / Search","cassandra"),
    Connector("redis","Redis","🔴",[F("host","Host",True),F("port","Port","int"),F("password","Password","password"),F("db","DB Index")],["password"],"NoSQL / Graph / Search","redis"),
    Connector("dynamodb","DynamoDB","🌀",[F("aws_access_key_id","AWS Access Key ID"),F("aws_secret_access_key","AWS Secret","password"),F("region_name","Region","text","ap-southeast-2")],["aws_secret_access_key"],"NoSQL / Graph / Search","dynamodb"),
    Connector("neo4j","Neo4j (Graph)","🕸️",[F("uri","Bolt URI",True,placeholder="bolt://localhost:7687"),F("user","User",True),F("password","Password",True,"password")],["password"],"NoSQL / Graph / Search","neo4j"),
    Connector("elasticsearch","Elasticsearch / OpenSearch","🔎",[F("hosts","Hosts",True,placeholder="http://localhost:9200,http://node2:9200"),F("username","Username"),F("password","Password","password")],["password"],"NoSQL / Graph / Search","elasticsearch"),
    Connector("cosmos","Azure Cosmos DB","🪐",[F("endpoint","Endpoint",True,placeholder="https://<acct>.documents.azure.com:443/"),F("key","Key",True,"password")],["key"],"NoSQL / Graph / Search","cosmos"),
    Connector("firestore","Firestore","🔥",[F("project_id","Project ID",True),F("credentials_json","Service Account JSON",True,"textarea")],["credentials_json"],"NoSQL / Graph / Search","firestore"),
    Connector("bigtable","Bigtable","📚",[F("project_id","Project ID",True),F("instance_id","Instance ID",True),F("credentials_json","Service Account JSON",True,"textarea")],["credentials_json"],"NoSQL / Graph / Search","bigtable"),
    Connector("s3","Amazon S3","🪣",[F("aws_access_key_id","AWS Access Key ID"),F("aws_secret_access_key","AWS Secret","password"),F("region_name","Region"),F("bucket","Bucket")],["aws_secret_access_key"],"Object Storage / Data Lake","s3"),
    Connector("azureblob","Azure Blob Storage","☁️",[F("connection_string","Connection String","text","DefaultEndpointsProtocol=..."),F("account_name","Account Name"),F("account_key","Account Key","password"),F("sas_url","Container SAS URL","text","https://.../container?...")],["connection_string","account_key","sas_url"],"Object Storage / Data Lake","azureblob"),
    Connector("adls","Azure Data Lake Gen2","📁",[F("account_name","Account Name"),F("account_key","Account Key","password"),F("filesystem","Filesystem/Container"),F("tenant_id","Tenant ID"),F("client_id","Client ID"),F("client_secret","Client Secret","password")],["account_key","client_secret"],"Object Storage / Data Lake","adls"),
    Connector("gcs","Google Cloud Storage","☁️",[F("project_id","Project ID"),F("bucket","Bucket"),F("credentials_json","Service Account JSON","textarea")],["credentials_json"],"Object Storage / Data Lake","gcs"),
    Connector("hdfs","HDFS (WebHDFS)","🗄️",[F("host","Host",True),F("port","Port","int"),F("user","User")],[],"Object Storage / Data Lake","hdfs"),
    Connector("kafka","Apache Kafka","📡",[F("bootstrap_servers","Bootstrap Servers",True,"text","host1:9092,host2:9092"),F("security_protocol","Security Protocol","select",options=["PLAINTEXT","SASL_PLAINTEXT","SASL_SSL","SSL"]),F("sasl_mechanism","SASL Mechanism","select",options=["","PLAIN","SCRAM-SHA-256","SCRAM-SHA-512"]),F("sasl_username","SASL Username"),F("sasl_password","SASL Password","password")],["sasl_password"],"Streaming / Messaging","kafka"),
    Connector("rabbitmq","RabbitMQ","🐇",[F("amqp_url","AMQP URL",True,"text","amqp://user:pass@host:5672/vhost")],["amqp_url"],"Streaming / Messaging","rabbitmq"),
    Connector("eventhubs","Azure Event Hubs","⚡",[F("connection_str","Connection String",True,"text","Endpoint=sb://...;SharedAccessKeyName=...;SharedAccessKey=..."),F("eventhub","Event Hub Name")],["connection_str"],"Streaming / Messaging","eventhubs"),
    Connector("pubsub","Google Pub/Sub","📣",[F("project_id","Project ID",True),F("credentials_json","Service Account JSON",True,"textarea")],["credentials_json"],"Streaming / Messaging","pubsub"),
    Connector("kinesis","AWS Kinesis","🌊",[F("aws_access_key_id","AWS Access Key ID"),F("aws_secret_access_key","AWS Secret","password"),F("region_name","Region")],["aws_secret_access_key"],"Streaming / Messaging","kinesis"),
    Connector("spark","Apache Spark","🔥",[F("master","Master URL",True,"text","local[*] or spark://host:7077"),F("app_name","App Name","text","ConnectorsHub")],[],"Big Data / Compute","spark"),
    Connector("dask","Dask","🐍",[F("scheduler_address","Scheduler Address","text","tcp://127.0.0.1:8786")],[],"Big Data / Compute","dask"),
    Connector("salesforce","Salesforce","☁️",[F("username","Username",True),F("password","Password",True,"password"),F("security_token","Security Token",True,"password"),F("domain","Domain","text","login or test")],["password","security_token"],"BI / SaaS","salesforce"),
    Connector("servicenow","ServiceNow","🧰",[F("instance","Instance",True,"text","dev12345"),F("user","User",True),F("password","Password",True,"password")],["password"],"BI / SaaS","servicenow"),
    Connector("jira","Jira","🧩",[F("server","Server URL",True,"text","https://yourdomain.atlassian.net"),F("email","Email",True),F("api_token","API Token",True,"password")],["api_token"],"BI / SaaS","jira"),
    Connector("sharepoint","SharePoint / Microsoft Graph","🗂️",[F("tenant_id","Tenant ID",True),F("client_id","Client ID",True),F("client_secret","Client Secret",True,"password")],["client_secret"],"BI / SaaS","sharepoint"),
    Connector("tableau","Tableau Server/Online","📊",[F("server","Server",True),F("site_id","Site ID"),F("token_name","Token Name",True),F("token_secret","Token Secret",True,"password")],["token_secret"],"BI / SaaS","tableau"),
    Connector("gmail","Gmail API","✉️",[F("credentials_json","OAuth Client/Service Account JSON",True,"textarea")],["credentials_json"],"Email / Collaboration","gmail"),
    Connector("msgraph","Microsoft Graph (Mail/Drive)","📧",[F("tenant_id","Tenant ID",True),F("client_id","Client ID",True),F("client_secret","Client Secret",True,"password")],["client_secret"],"Email / Collaboration","msgraph"),
]
REG_BY_ID: Dict[str, Connector] = {c.id: c for c in REGISTRY}

# ---------------------- Sidebar (LHS) ----------------------
def _sorted_filtered_connectors(q: str) -> List[Connector]:
    items = sorted(REGISTRY, key=lambda x: x.name.lower())
    if not q:
        return items
    ql = q.lower()
    return [c for c in items if ql in c.name.lower() or ql in c.id.lower() or ql in c.category.lower()]

def _get_query_params() -> Dict[str, List[str]]:
    try:
        return dict(st.query_params)  # Streamlit >=1.30
    except Exception:
        return st.experimental_get_query_params()

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

    qp = _get_query_params()
    selected_id = (qp.get("conn")[0] if isinstance(qp.get("conn"), list) else qp.get("conn")) if qp.get("conn") else None
    if selected_id not in {c.id for c in REGISTRY}:
        selected_id = (filtered[0].id if filtered else REGISTRY[0].id)

    # Clicking any item opens the RHS panel via panel=profile
    rows_html = []
    for c in filtered:
        active = "is-active" if c.id == selected_id and (qp.get("panel", [""])[0] == "profile") else ""
        rows_html.append(
            f'<a class="sidebar-row {active}" href="?conn={c.id}&panel=profile" target="_self" title="{c.name}">'
            f'<span class="sidebar-ico">{c.icon}</span><span class="sidebar-name">{c.name}</span></a>'
        )

    st.markdown(
        dedent(
            f"""
            <div class="sidebar-wrap">
              <div class="sidebar-scroll">
                {''.join(rows_html)}
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

# Resolve current connector and panel mode
conn: Connector = REG_BY_ID[selected_id]  # type: ignore[index]
qp = _get_query_params()
panel_mode = (qp.get("panel")[0] if isinstance(qp.get("panel"), list) else qp.get("panel")) if qp.get("panel") else ""

# ---------------------- Load / Save storage ----------------------
all_profiles = _load_all()
profiles_for = all_profiles.get(conn.id, {})

# ---------------------- Header / Overview ----------------------
k1, k2, k3 = st.columns([2,1,1])
with k1:
    thumb = _logo_html(conn.logo_key or conn.id, size=22)
    if thumb:
        st.markdown(f'<div class="logo-wrap">{thumb}<h2 style="margin:0;">{conn.icon} {conn.name}</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown(f"## {conn.icon} {conn.name}")
    st.caption(f"Category: **{conn.category}** · ID: `{conn.id}`")
with k2:
    st.markdown(f'<div class="kpi">🧩 Connectors: <b>{len(REGISTRY)}</b></div>', unsafe_allow_html=True)
with k3:
    total_profiles = sum(len(v) for v in all_profiles.values())
    st.markdown(f'<div class="kpi">🗂️ Profiles: <b>{total_profiles}</b></div>', unsafe_allow_html=True)

st.write("")

# ---------------------- Columns: main + RHS panel space ----------------------
main_left, main_right = st.columns([7, 5], gap="large")

# ---------------------- Test handlers (ALL) ----------------------
# (Handlers are long and unchanged; for completeness they are identical to earlier message)
# -- BEGIN handlers block --
#  (omitted here to keep this message focused; use the exact handlers block from earlier code)
# -- END handlers block --

# --------------- Reusable: render Configure form into any container ---------------
def render_configure_form(container, conn: Connector):
    with container:
        st.markdown("#### Configure connection profile")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        _keys = _get_state_keys(conn.id)
        for k in _keys.values():
            st.session_state.setdefault(k, None)

        with st.form(key=f"form_{conn.id}", clear_on_submit=False):
            profile_name = st.text_input("Profile Name", placeholder="dev / staging / prod", key=f"{conn.id}_profile_name")

            values: Dict[str, Any] = {}
            missing_required: List[str] = []
            for f in conn.fields:
                ikey = f"{conn.id}_{f.key}"
                if f.kind == "password":
                    val = st.text_input(f.label, type="password", placeholder=f.placeholder or "", key=ikey)
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
                if f.required and (val is None or str(val).strip() == ""):
                    missing_required.append(f.label)

            c1, c2, c3, c4 = st.columns([1,1,1,1])
            submitted = c1.form_submit_button("💾 Save Profile", use_container_width=True)
            preview   = c2.form_submit_button("🧪 Preview DSN", use_container_width=True)
            envvars   = c3.form_submit_button("🔐 Env-Vars Snippet", use_container_width=True)
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
                    ok, msg = handler(values)
                st.session_state[_keys["ok"]] = bool(ok)
                st.session_state[_keys["msg"]] = msg
                st.session_state[_keys["sig"]] = _config_signature(values)

        last_ok  = st.session_state.get(_keys["ok"])
        last_msg = st.session_state.get(_keys["msg"]) or "Not tested"
        sig_now  = _config_signature(values)
        sig_last = st.session_state.get(_keys["sig"])

        if last_ok is True and sig_now == sig_last:
            st.markdown(f'<span class="pill">✅ Successful</span> <span class="muted small">{last_msg}</span>', unsafe_allow_html=True)
        elif last_ok is False and sig_now == sig_last:
            st.markdown(f'<span class="pill" style="background:#FEE2E2;color:#991B1B;">❌ Failed</span> <span class="muted small">{last_msg}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill" style="background:#FFF7ED;color:#9A3412;">⏳ Not tested</span> <span class="muted small">Run “Test Connection” before saving.</span>', unsafe_allow_html=True)

        if preview:
            st.info("Indicative DSN/URI preview (no network calls):")
            st.code(_dsn_preview(conn.id, values), language="text")

        if envvars:
            st.info("Copy/paste into your shell (masked preview):")
            st.code(_env_snippet(conn.id, profile_name or "PROFILE", values, conn.secret_keys), language="bash")

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

# ---------------------- Saved Profiles (reusable) ----------------------
def render_saved_profiles(container, conn: Connector, profiles_for: Dict[str, Dict[str, Any]]):
    with container:
        st.markdown("#### Saved profiles")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if not profiles_for:
            st.info("No profiles saved yet for this connector.")
        else:
            for pname, cfg in sorted(profiles_for.items()):
                redacted = masked_view(cfg, conn.secret_keys)
                with st.expander(f"{conn.icon} {conn.name} — **{pname}**", expanded=False):
                    st.json(redacted)
                    cc1, cc2, cc3, cc4 = st.columns([1,1,1,2])
                    if cc1.button("📝 Load to form", key=f"load_{conn.id}_{pname}"):
                        st.session_state[f"{conn.id}_profile_name"] = pname
                        for f in conn.fields:
                            st.session_state[f"{conn.id}_{f.key}"] = cfg.get(f.key, "")
                        st.success(f"Loaded **{pname}** into the form.")
                    if cc2.button("🧪 Test", key=f"quicktest_{conn.id}_{pname}"):
                        handler = TEST_HANDLERS.get(conn.id)
                        if handler:
                            _short_timeout_env()
                            with st.spinner("Testing connection..."):
                                ok, msg = handler(cfg)
                            st.success(msg) if ok else st.error(msg)
                            _cache_status_set(conn.id, pname, bool(ok), msg)
                        else:
                            st.warning("No test implemented.")
                    if cc3.button("🗑️ Delete", key=f"del_{conn.id}_{pname}"):
                        all_profiles = _load_all()
                        all_profiles.get(conn.id, {}).pop(pname, None)
                        if not all_profiles.get(conn.id):
                            all_profiles.pop(conn.id, None)
                        _save_all(all_profiles)
                        st.session_state.pop(_status_cache_key(conn.id, pname), None)
                        st.warning(f"Deleted profile **{pname}**.")
                        st.rerun()
                    cc4.caption("Secrets are masked in this view. Raw values remain local in `connections.json`.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Conditional layout ----------------------
if panel_mode == "profile":
    # RHS panel shows the Configure form; main area shows Saved profiles
    with main_right:
        st.markdown('<div class="rhs-aside">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rhs-header"><div><b>Configure connection profile</b></div>'
            f'<div class="rhs-close"><a href="?conn={conn.id}" target="_self">✖ Close panel</a></div></div>',
            unsafe_allow_html=True,
        )
    render_configure_form(main_right, conn)
    with main_right:
        st.markdown('</div>', unsafe_allow_html=True)

    render_saved_profiles(main_left, conn, profiles_for)

else:
    # Default: form in main_left; saved profiles in main_right (original layout)
    render_configure_form(main_left, conn)
    render_saved_profiles(main_right, conn, profiles_for)

# ---------------------- All Configured Connections (Table) ----------------------
st.write("")
st.markdown("### 📚 All configured connections")
st.markdown('<div class="card">', unsafe_allow_html=True)

def _run_status_check_for_all():
    _short_timeout_env()
    for cid, items in (_load_all() or {}).items():
        handler = TEST_HANDLERS.get(cid)
        for pname, cfg in (items or {}).items():
            if handler:
                try: ok, msg = handler(cfg)
                except Exception as e: ok, msg = False, str(e)
            else:
                ok, msg = None, "No test implemented."
            _cache_status_set(cid, pname, ok, msg)

cA, cB = st.columns([1,3])
with cA:
    if st.button("🔁 Test all saved connections now", use_container_width=True):
        with st.spinner("Running status checks..."):
            _run_status_check_for_all()

all_profiles = _load_all()  # re-read after potential changes above
if not all_profiles:
    st.info("You haven’t saved any connections yet.")
else:
    rows: List[Dict[str, Any]] = []
    for cid, items in all_profiles.items():
        meta = REG_BY_ID.get(cid)
        for pname, cfg in items.items():
            status_info = _cache_status_get(cid, pname)
            ok = status_info.get("ok")
            msg = status_info.get("msg") or ""
            status_text = "✅ Successful" if ok is True else ("❌ Failed" if ok is False else "⏳ Not tested")
            rows.append({
                "Connector": f"{meta.icon if meta else '🔌'} {meta.name if meta else cid}",
                "Profile": pname,
                "Fields": ", ".join(cfg.keys()),
                "Status": status_text,
                "Details": msg,
            })
    df = pd.DataFrame(rows).sort_values(["Connector", "Profile"])
    st.dataframe(df, hide_index=True, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Import / Export JSON ----------------------
st.write("")
st.markdown("### 🔄 Import / Export")
cA, cB = st.columns([1,1])

with cA:
    st.markdown("#### Export connections")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if all_profiles:
        export_json = json.dumps(all_profiles, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download connections.json",
            data=export_json,
            file_name="connections.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Nothing to export yet.")
    st.markdown('</div>', unsafe_allow_html=True)

with cB:
    st.markdown("#### Import connections")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    up = st.file_uploader("Upload a connections.json", type=["json"])
    if up:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict):
                merged = _load_all()
                for cid, profs in data.items():
                    merged.setdefault(cid, {})
                    merged[cid].update(profs or {})
                _save_all(merged)
                st.success("Imported connections successfully.")
                st.rerun()
            else:
                st.error("Invalid format. Expected a JSON object.")
        except Exception as e:
            st.error(f"Failed to import: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
