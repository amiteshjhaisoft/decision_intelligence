# Author: Amitesh Jha | iSoft | 2025-10-12
# connectors_hub.py
# Professional Connectors Hub (Streamlit)

from __future__ import annotations

import base64
import json
import os
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
PIPE_STORE = APP_DIR / "pipelines.json"   # <-- add this line
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

      /* small header row inside configure card */
      .card-title-row{display:flex;align-items:center;justify-content:space-between;margin:0 0 .5rem;}
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
        if conn_id == "weaviate":
            # Prefer a full cluster URL (WCS); otherwise build from host/port.
            cluster_url = (cfg.get("cluster_url") or cfg.get("url") or "").strip().rstrip("/")
        
            if cluster_url:
                http_url = cluster_url
                scheme = "https" if cluster_url.startswith("https") else "http"
            else:
                scheme = (cfg.get("scheme") or "https").lower()
                host = (cfg.get("host") or "").strip()
                port = int(cfg.get("port") or (443 if scheme == "https" else 80))
                default_port = 443 if scheme == "https" else 80
                http_url = f"{scheme}://{host}" + ("" if port == default_port else f":{port}")
        
            mt = " (multi-tenancy)" if str(cfg.get("multi_tenancy") or "").lower() in ("true", "yes", "1") else ""
        
            # If you later add gRPC fields, uncomment to display them in the preview.
            # grpc_host = (cfg.get("grpc_host") or host or "").strip()
            # grpc_port = cfg.get("grpc_port")
            # grpc_part = f" [grpc={grpc_host}:{grpc_port}]" if grpc_host and grpc_port else ""
        
            return f"weaviate://{http_url}{mt}"  # + grpc_part
  
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

REGISTRY: List[Connector] = [
    # --- SQL ---
    Connector("postgres", "PostgreSQL", "🐘",
              [F("host","Host",required=True), F("port","Port",kind="int"),
               F("database","Database",required=True), F("user","User",required=True),
               F("password","Password",required=True,kind="password")],
              ["password"], "SQL", "postgres"),
    Connector("mysql", "MySQL / MariaDB", "🐬",
              [F("host","Host",required=True), F("port","Port",kind="int"),
               F("database","Database",required=True), F("user","User",required=True),
               F("password","Password",required=True,kind="password")],
              ["password"], "SQL", "mysql"),
    Connector("mssql", "SQL Server / Azure SQL", "🪟",
              [F("driver","ODBC Driver",placeholder="ODBC Driver 18 for SQL Server"),
               F("server","Server",required=True), F("database","Database",required=True),
               F("user","User",required=True), F("password","Password",required=True,kind="password")],
              ["password"], "SQL", "mssql"),
    Connector("oracle", "Oracle", "🏺",
              [F("dsn","DSN",required=True,placeholder="host:port/service_name"),
               F("user","User",required=True), F("password","Password",required=True,kind="password")],
              ["password"], "SQL", "oracle"),
    Connector("sqlite", "SQLite", "🗂️",
              [F("filepath","DB File Path",required=True,placeholder="./my.db")],
              [], "SQL", "sqlite"),
    Connector("trino", "Trino / Presto", "🚀",
              [F("host","Host",required=True), F("port","Port",kind="int"),
               F("catalog","Catalog",required=True), F("schema","Schema",required=True),
               F("user","User",required=True)],
              [], "SQL", "trino"),
    Connector("duckdb", "DuckDB", "🦆",
              [F("filepath","DB File Path",required=True,placeholder="./warehouse.duckdb")],
              [], "SQL", "duckdb"),

    # --- Cloud DW / Analytics ---
    Connector("snowflake","Snowflake","❄️",
              [F("account","Account",required=True,placeholder="xy12345.ap-southeast-2"),
               F("user","User",required=True), F("password","Password",required=True,kind="password"),
               F("warehouse","Warehouse"), F("database","Database"),
               F("schema","Schema"), F("role","Role")],
              ["password"], "Cloud DW / Analytics", "snowflake"),
    Connector("bigquery","BigQuery","🧮",
              [F("project_id","Project ID",required=True),
               F("credentials_json","Service Account JSON",required=True,kind="textarea",placeholder="{...}")],
              ["credentials_json"], "Cloud DW / Analytics", "bigquery"),
    Connector("redshift","Amazon Redshift","🧊",
              [F("host","Host",required=True), F("port","Port",kind="int"),
               F("database","Database",required=True), F("user","User",required=True),
               F("password","Password",required=True,kind="password")],
              ["password"], "Cloud DW / Analytics", "redshift"),
    Connector("synapse","Azure Synapse (SQL)","🔷",
              [F("server","Server",required=True,placeholder="yourserver.database.windows.net"),
               F("database","Database",required=True), F("user","User",required=True),
               F("password","Password",required=True,kind="password")],
              ["password"], "Cloud DW / Analytics", "synapse"),

    # --- NoSQL / Graph / Search ---
    Connector("mongodb","MongoDB","🍃",
              [F("uri","Mongo URI",required=True,placeholder="mongodb+srv://user:pass@cluster/db")],
              ["uri"], "NoSQL / Graph / Search", "mongodb"),
    Connector("cassandra","Cassandra","💠",
              [F("contact_points","Contact Points",required=True,placeholder="host1,host2"),
               F("port","Port",kind="int"), F("username","Username"),
               F("password","Password",kind="password"), F("keyspace","Keyspace")],
              ["password"], "NoSQL / Graph / Search", "cassandra"),
    Connector("redis","Redis","🔴",
              [F("host","Host",required=True), F("port","Port",kind="int"),
               F("password","Password",kind="password"), F("db","DB Index")],
              ["password"], "NoSQL / Graph / Search", "redis"),
    Connector("dynamodb","DynamoDB","🌀",
              [F("aws_access_key_id","AWS Access Key ID"),
               F("aws_secret_access_key","AWS Secret",kind="password"),
               F("region_name","Region",placeholder="ap-southeast-2")],
              ["aws_secret_access_key"], "NoSQL / Graph / Search", "dynamodb"),
    Connector("neo4j","Neo4j (Graph)","🕸️",
              [F("uri","Bolt URI",required=True,placeholder="bolt://localhost:7687"),
               F("user","User",required=True), F("password","Password",required=True,kind="password")],
              ["password"], "NoSQL / Graph / Search", "neo4j"),
    Connector("elasticsearch","Elasticsearch / OpenSearch","🔎",
              [F("hosts","Hosts",required=True,placeholder="http://localhost:9200,http://node2:9200"),
               F("username","Username"), F("password","Password",kind="password")],
              ["password"], "NoSQL / Graph / Search", "elasticsearch"),
    Connector("cosmos","Azure Cosmos DB","🪐",
              [F("endpoint","Endpoint",required=True,placeholder="https://<acct>.documents.azure.com:443/"),
               F("key","Key",required=True,kind="password")],
              ["key"], "NoSQL / Graph / Search", "cosmos"),
    Connector("firestore","Firestore","🔥",
              [F("project_id","Project ID",required=True),
               F("credentials_json","Service Account JSON",required=True,kind="textarea")],
              ["credentials_json"], "NoSQL / Graph / Search", "firestore"),
    Connector("bigtable","Bigtable","📚",
              [F("project_id","Project ID",required=True), F("instance_id","Instance ID",required=True),
               F("credentials_json","Service Account JSON",required=True,kind="textarea")],
              ["credentials_json"], "NoSQL / Graph / Search", "bigtable"),

    # --- Object Storage / Data Lake ---
    Connector("s3","Amazon S3","🪣",
              [F("aws_access_key_id","AWS Access Key ID"),
               F("aws_secret_access_key","AWS Secret",kind="password"),
               F("region_name","Region"), F("bucket","Bucket")],
              ["aws_secret_access_key"], "Object Storage / Data Lake", "s3"),
    Connector("azureblob","Azure Blob Storage","☁️",
              [F("connection_string","Connection String",placeholder="DefaultEndpointsProtocol=..."),
               F("account_name","Account Name"), F("account_key","Account Key",kind="password"),
               F("sas_url","Container SAS URL",placeholder="https://.../container?...")],
              ["connection_string","account_key","sas_url"], "Object Storage / Data Lake", "azureblob"),
    Connector("adls","Azure Data Lake Gen2","📁",
              [F("account_name","Account Name"), F("account_key","Account Key",kind="password"),
               F("filesystem","Filesystem/Container"), F("tenant_id","Tenant ID"),
               F("client_id","Client ID"), F("client_secret","Client Secret",kind="password")],
              ["account_key","client_secret"], "Object Storage / Data Lake", "adls"),
    Connector("gcs","Google Cloud Storage","☁️",
              [F("project_id","Project ID"), F("bucket","Bucket"),
               F("credentials_json","Service Account JSON",kind="textarea")],
              ["credentials_json"], "Object Storage / Data Lake", "gcs"),
    Connector("hdfs","HDFS (WebHDFS)","🗄️",
              [F("host","Host",required=True), F("port","Port",kind="int"), F("user","User")],
              [], "Object Storage / Data Lake", "hdfs"),

    # --- Streaming / Messaging ---
    Connector("kafka","Apache Kafka","📡",
              [F("bootstrap_servers","Bootstrap Servers",required=True,placeholder="host1:9092,host2:9092"),
               F("security_protocol","Security Protocol",kind="select",
                 options=["PLAINTEXT","SASL_PLAINTEXT","SASL_SSL","SSL"]),
               F("sasl_mechanism","SASL Mechanism",kind="select",
                 options=["","PLAIN","SCRAM-SHA-256","SCRAM-SHA-512"]),
               F("sasl_username","SASL Username"),
               F("sasl_password","SASL Password",kind="password")],
              ["sasl_password"], "Streaming / Messaging", "kafka"),
    Connector("rabbitmq","RabbitMQ","🐇",
              [F("amqp_url","AMQP URL",required=True,placeholder="amqp://user:pass@host:5672/vhost")],
              ["amqp_url"], "Streaming / Messaging", "rabbitmq"),
    Connector("eventhubs","Azure Event Hubs","⚡",
              [F("connection_str","Connection String",required=True,placeholder="Endpoint=sb://...;SharedAccessKeyName=...;SharedAccessKey=..."),
               F("eventhub","Event Hub Name")],
              ["connection_str"], "Streaming / Messaging", "eventhubs"),
    Connector("pubsub","Google Pub/Sub","📣",
              [F("project_id","Project ID",required=True),
               F("credentials_json","Service Account JSON",required=True,kind="textarea")],
              ["credentials_json"], "Streaming / Messaging", "pubsub"),
    Connector("kinesis","AWS Kinesis","🌊",
              [F("aws_access_key_id","AWS Access Key ID"),
               F("aws_secret_access_key","AWS Secret",kind="password"),
               F("region_name","Region")],
              ["aws_secret_access_key"], "Streaming / Messaging", "kinesis"),

    # --- Big Data / Compute ---
    Connector("spark","Apache Spark","🔥",
              [F("master","Master URL",required=True,placeholder="local[*] or spark://host:7077"),
               F("app_name","App Name",placeholder="ConnectorsHub")],
              [], "Big Data / Compute", "spark"),
    Connector("dask","Dask","🐍",
              [F("scheduler_address","Scheduler Address",placeholder="tcp://127.0.0.1:8786")],
              [], "Big Data / Compute", "dask"),

    # --- BI / SaaS ---
    Connector("salesforce","Salesforce","☁️",
              [F("username","Username",required=True),
               F("password","Password",required=True,kind="password"),
               F("security_token","Security Token",required=True,kind="password"),
               F("domain","Domain",placeholder="login or test")],
              ["password","security_token"], "BI / SaaS", "salesforce"),
    Connector("servicenow","ServiceNow","🧰",
              [F("instance","Instance",required=True,placeholder="dev12345"),
               F("user","User",required=True), F("password","Password",required=True,kind="password")],
              ["password"], "BI / SaaS", "servicenow"),
    Connector("jira","Jira","🧩",
              [F("server","Server URL",required=True,placeholder="https://yourdomain.atlassian.net"),
               F("email","Email",required=True), F("api_token","API Token",required=True,kind="password")],
              ["api_token"], "BI / SaaS", "jira"),
    Connector("sharepoint","SharePoint / Microsoft Graph","🗂️",
              [F("tenant_id","Tenant ID",required=True), F("client_id","Client ID",required=True),
               F("client_secret","Client Secret",required=True,kind="password")],
              ["client_secret"], "BI / SaaS", "sharepoint"),
    Connector("tableau","Tableau Server/Online","📊",
              [F("server","Server",required=True), F("site_id","Site ID"), F("token_name","Token Name",required=True),
               F("token_secret","Token Secret",required=True,kind="password")],
              ["token_secret"], "BI / SaaS", "tableau"),

    # --- Email / Collaboration ---
    Connector("gmail","Gmail API","✉️",
              [F("credentials_json","OAuth Client/Service Account JSON",required=True,kind="textarea")],
              ["credentials_json"], "Email / Collaboration", "gmail"),
    Connector("msgraph","Microsoft Graph (Mail/Drive)","📧",
              [F("tenant_id","Tenant ID",required=True), F("client_id","Client ID",required=True),
               F("client_secret","Client Secret",required=True,kind="password")],
              ["client_secret"], "Email / Collaboration", "msgraph"),
    Connector("weaviate", "Weaviate (Vector DB)", "🧠",
              [F("url","Cluster URL", required=False, placeholder="https://your-cluster.weaviate.network"),
               F("scheme","Scheme", kind="select", options=["https","http"]),
               F("host","Host", placeholder="your-cluster.weaviate.network"),
               F("port","Port", kind="int"),
               F("api_key","API Key", kind="password"),
               F("multi_tenancy","Multi-Tenancy", kind="select", options=["", "true", "false"])
              ],
              ["api_key"],
              "NoSQL / Graph / Search",
              "weaviate"),

]

REG_BY_ID: Dict[str, Connector] = {c.id: c for c in REGISTRY}

# ---------------------- Sidebar (SPA navigation: no hard refresh) ----------------------
def _sorted_filtered_connectors(q: str) -> List[Connector]:
    items = sorted(REGISTRY, key=lambda x: x.name.lower())
    if not q:
        return items
    ql = q.lower()
    return [c for c in items if ql in c.name.lower() or ql in c.id.lower() or ql in c.category.lower()]

# Session state defaults
st.session_state.setdefault("selected_id", REGISTRY[0].id)
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
    for c in filtered:
        is_active = (st.session_state["selected_id"] == c.id and st.session_state["rhs_open"])
        if st.button(f"{c.icon}  {c.name}", key=f"nav_{c.id}",
                     type=("primary" if is_active else "secondary"),
                     use_container_width=True):
            _set_active(c.id)

    # ---------- Import / Export moved into the sidebar ----------
    st.divider()
    with st.expander("🔄 Import / Export Connections", expanded=False):
        sidebar_profiles = _load_all()

        # Export
        st.markdown("**Export connections**")
        if sidebar_profiles:
            export_json = json.dumps(sidebar_profiles, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download connections.json",
                data=export_json,
                file_name="connections.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Nothing to export yet.")

        st.write("")

        # Import
        st.markdown("**Import connections**")
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

# Resolve current connector from state
conn: Connector = REG_BY_ID[st.session_state["selected_id"]]

# # ---------------------- Load store & KPIs ----------------------
all_profiles = _load_all()
total_profiles_all = sum(len(v) for v in all_profiles.values())

# (Keep the red-circled KPIs)
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
    # prefill profile name and all fields for this connector
    st.session_state[f"{conn_id}_profile_name"] = profile
    for f in REG_BY_ID[conn_id].fields:
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

# ---------------------- Test handlers (per connector) ----------------------
# (All handlers remain unchanged; omitted comments for brevity.)
def test_postgres(cfg):
    try:
        import psycopg2
        dsn = {
            "host": cfg.get("host"),
            "port": int(cfg.get("port") or 5432),
            "dbname": cfg.get("database"),
            "user": cfg.get("user"),
            "password": cfg.get("password"),
            "connect_timeout": 5,
        }
        conn_ = psycopg2.connect(**dsn)
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone()
        conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install psycopg2-binary"
    except Exception as e:
        return False, f"{e}"

def test_mysql(cfg):
    try:
        import pymysql
        conn_ = pymysql.connect(host=cfg.get("host"), user=cfg.get("user"),
                                password=cfg.get("password"), database=cfg.get("database"),
                                port=int(cfg.get("port") or 3306), connect_timeout=5)
        with conn_.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install PyMySQL"
    except Exception as e:
        return False, f"{e}"

def test_mssql(cfg):
    try:
        import pyodbc
        driver = cfg.get("driver") or "ODBC Driver 18 for SQL Server"
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={cfg.get('server')};DATABASE={cfg.get('database')};"
            f"UID={cfg.get('user')};PWD={cfg.get('password')};TrustServerCertificate=yes;Connection Timeout=5;"
        )
        conn_ = pyodbc.connect(conn_str)
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone()
        conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install: ODBC Driver + pip install pyodbc"
    except Exception as e:
        return False, f"{e}"

def test_oracle(cfg):
    try:
        import oracledb
        conn_ = oracledb.connect(user=cfg.get("user"), password=cfg.get("password"), dsn=cfg.get("dsn"), timeout=5)
        cur = conn_.cursor(); cur.execute("SELECT 1 FROM dual"); cur.fetchone()
        conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install oracledb"
    except Exception as e:
        return False, f"{e}"

def test_sqlite(cfg):
    try:
        import sqlite3
        conn_ = sqlite3.connect(cfg.get("filepath"))
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone()
        conn_.close()
        return True, "Opened DB file successfully."
    except Exception as e:
        return False, f"{e}"

def test_trino(cfg):
    try:
        import trino
        conn_ = trino.dbapi.connect(host=cfg.get("host"), port=int(cfg.get("port") or 8080),
                                    user=cfg.get("user"), catalog=cfg.get("catalog"),
                                    schema=cfg.get("schema"))
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone()
        conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install trino"
    except Exception as e:
        return False, f"{e}"

def test_duckdb(cfg):
    try:
        import duckdb
        conn_ = duckdb.connect(cfg.get("filepath"))
        conn_.execute("SELECT 1").fetchone()
        conn_.close()
        return True, "Opened DuckDB file successfully."
    except ModuleNotFoundError:
        return False, "Install library: pip install duckdb"
    except Exception as e:
        return False, f"{e}"

def test_snowflake(cfg):
    try:
        import snowflake.connector as sf
        conn_ = sf.connect(user=cfg.get("user"), password=cfg.get("password"),
                           account=cfg.get("account"), warehouse=cfg.get("warehouse"),
                           database=cfg.get("database"), schema=cfg.get("schema"), role=cfg.get("role"))
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone(); cur.close(); conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install snowflake-connector-python"
    except Exception as e:
        return False, f"{e}"

def _google_creds_from_json(txt):
    import json as _json
    from google.oauth2 import service_account
    info = _json.loads(txt)
    return service_account.Credentials.from_service_account_info(info)

def test_bigquery(cfg):
    try:
        from google.cloud import bigquery
        creds = _google_creds_from_json(cfg.get("credentials_json"))
        client = bigquery.Client(project=cfg.get("project_id"), credentials=creds)
        list(client.list_datasets(page_size=1))
        return True, "Client initialized; datasets listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install google-cloud-bigquery"
    except Exception as e:
        return False, f"{e}"

def test_redshift(cfg):
    try:
        import psycopg2
        conn_ = psycopg2.connect(host=cfg.get("host"), port=int(cfg.get("port") or 5439),
                                 dbname=cfg.get("database"), user=cfg.get("user"),
                                 password=cfg.get("password"), connect_timeout=5)
        cur = conn_.cursor(); cur.execute("SELECT 1"); cur.fetchone(); conn_.close()
        return True, "Connected (SELECT 1 ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install psycopg2-binary"
    except Exception as e:
        return False, f"{e}"

def test_synapse(cfg):
    return test_mssql(cfg)

def test_mongodb(cfg):
    try:
        from pymongo import MongoClient
        cli = MongoClient(cfg.get("uri"), serverSelectionTimeoutMS=5000)
        cli.admin.command("ping")
        cli.close()
        return True, "Connected (ping ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install pymongo"
    except Exception as e:
        return False, f"{e}"

def test_cassandra(cfg):
    try:
        from cassandra.cluster import Cluster
        cps = [h.strip() for h in (cfg.get("contact_points") or "").split(",") if h.strip()]
        cluster = Cluster(contact_points=cps, port=int(cfg.get("port") or 9042))
        session = cluster.connect(timeout=5)
        session.shutdown(); cluster.shutdown()
        return True, "Connected (session established)."
    except ModuleNotFoundError:
        return False, "Install library: pip install cassandra-driver"
    except Exception as e:
        return False, f"{e}"

def test_redis(cfg):
    try:
        import redis
        r = redis.Redis(host=cfg.get("host"), port=int(cfg.get("port") or 6379),
                        password=cfg.get("password"), db=int(cfg.get("db") or 0), socket_timeout=5)
        r.ping()
        return True, "Connected (PING ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install redis"
    except Exception as e:
        return False, f"{e}"

def test_dynamodb(cfg):
    try:
        import boto3
        client = boto3.client("dynamodb",
                              region_name=cfg.get("region_name"),
                              aws_access_key_id=cfg.get("aws_access_key_id"),
                              aws_secret_access_key=cfg.get("aws_secret_access_key"))
        client.list_tables(Limit=1)
        return True, "Client initialized; tables listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install boto3"
    except Exception as e:
        return False, f"{e}"

def test_neo4j(cfg):
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(cfg.get("uri"), auth=(cfg.get("user"), cfg.get("password")))
        drv.verify_connectivity()
        drv.close()
        return True, "Connected (verify_connectivity ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install neo4j"
    except Exception as e:
        return False, f"{e}"

def test_elasticsearch(cfg):
    try:
        from elasticsearch import Elasticsearch
        hosts = [h.strip() for h in (cfg.get("hosts") or "").split(",") if h.strip()]
        es = Elasticsearch(hosts, basic_auth=(cfg.get("username"), cfg.get("password")) if cfg.get("username") else None,
                           request_timeout=5)
        es.info()
        return True, "Connected (info ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install elasticsearch"
    except Exception as e:
        return False, f"{e}"

def test_cosmos(cfg):
    try:
        from azure.cosmos import CosmosClient
        client = CosmosClient(url=cfg.get("endpoint"), credential=cfg.get("key"))
        list(client.list_databases())
        return True, "Client initialized; databases listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install azure-cosmos"
    except Exception as e:
        return False, f"{e}"

def test_firestore(cfg):
    try:
        from google.cloud import firestore
        creds = _google_creds_from_json(cfg.get("credentials_json"))
        client = firestore.Client(project=cfg.get("project_id"), credentials=creds)
        _ = list(client.collections())[:1]
        return True, "Client initialized; collections accessible."
    except ModuleNotFoundError:
        return False, "Install library: pip install google-cloud-firestore"
    except Exception as e:
        return False, f"{e}"

def test_bigtable(cfg):
    try:
        from google.cloud import bigtable
        creds = _google_creds_from_json(cfg.get("credentials_json"))
        client = bigtable.Client(project=cfg.get("project_id"), credentials=creds, admin=True)
        instance = client.instance(cfg.get("instance_id"))
        instance.exists()
        return True, "Client initialized; instance checked."
    except ModuleNotFoundError:
        return False, "Install library: pip install google-cloud-bigtable"
    except Exception as e:
        return False, f"{e}"

def test_s3(cfg):
    try:
        import boto3
        client = boto3.client("s3",
                              region_name=cfg.get("region_name"),
                              aws_access_key_id=cfg.get("aws_access_key_id"),
                              aws_secret_access_key=cfg.get("aws_secret_access_key"))
        client.list_buckets()
        return True, "Client initialized; buckets listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install boto3 s3fs"
    except Exception as e:
        return False, f"{e}"

def test_azureblob(cfg):
    try:
        from azure.storage.blob import BlobServiceClient
        if cfg.get("connection_string"):
            svc = BlobServiceClient.from_connection_string(cfg.get("connection_string"))
        elif cfg.get("sas_url"):
            svc = BlobServiceClient(account_url=cfg.get("sas_url").split("?")[0], credential=cfg.get("sas_url"))
        elif cfg.get("account_name") and cfg.get("account_key"):
            acct = cfg.get("account_name")
            url = f"https://{acct}.blob.core.windows.net"
            svc = BlobServiceClient(account_url=url, credential=cfg.get("account_key"))
        else:
            return False, "Provide connection_string OR (account_name + account_key) OR sas_url."
        next(iter(svc.list_containers()), None)
        return True, "Client initialized; containers listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install azure-storage-blob"
    except Exception as e:
        return False, f"{e}"

def test_adls(cfg):
    try:
        from azure.storage.filedatalake import DataLakeServiceClient
        if cfg.get("account_name") and cfg.get("account_key"):
            url = f"https://{cfg.get('account_name')}.dfs.core.windows.net"
            svc = DataLakeServiceClient(account_url=url, credential=cfg.get("account_key"))
        else:
            return False, "Provide account_name + account_key."
        next(iter(svc.list_file_systems()), None)
        return True, "Client initialized; filesystems listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install azure-storage-file-datalake"
    except Exception as e:
        return False, f"{e}"

def test_gcs(cfg):
    try:
        from google.cloud import storage
        creds = None
        if cfg.get("credentials_json"):
            creds = _google_creds_from_json(cfg.get("credentials_json"))
        client = storage.Client(project=cfg.get("project_id"), credentials=creds)
        list(client.list_buckets(page_size=1))
        return True, "Client initialized; buckets listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install google-cloud-storage gcsfs"
    except Exception as e:
        return False, f"{e}"

def test_hdfs(cfg):
    try:
        from hdfs import InsecureClient
        url = f"http://{cfg.get('host')}:{int(cfg.get('port') or 9870)}"
        client = InsecureClient(url, user=cfg.get("user") or None, timeout=5)
        client.status("/", strict=False)
        return True, "Client initialized; root status retrieved."
    except ModuleNotFoundError:
        return False, "Install library: pip install hdfs"
    except Exception as e:
        return False, f"{e}"

def test_kafka(cfg):
    try:
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers=cfg.get("bootstrap_servers"),
                                 security_protocol=cfg.get("security_protocol") or "PLAINTEXT",
                                 sasl_mechanism=(cfg.get("sasl_mechanism") or None) or None,
                                 sasl_plain_username=(cfg.get("sasl_username") or None),
                                 sasl_plain_password=(cfg.get("sasl_password") or None),
                                 request_timeout_ms=5000, api_version_auto_timeout_ms=5000)
        admin.list_topics()
        admin.close()
        return True, "Connected (metadata ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install kafka-python"
    except Exception as e:
        return False, f"{e}"

def test_rabbitmq(cfg):
    try:
        import pika
        params = pika.URLParameters(cfg.get("amqp_url"))
        params.socket_timeout = 5
        conn_ = pika.BlockingConnection(params)
        conn_.close()
        return True, "Connected (AMQP ok)."
    except ModuleNotFoundError:
        return False, "Install library: pip install pika"
    except Exception as e:
        return False, f"{e}"

def test_eventhubs(cfg):
    try:
        from azure.eventhub import EventHubProducerClient
        client = EventHubProducerClient.from_connection_string(conn_str=cfg.get("connection_str"),
                                                               eventhub_name=cfg.get("eventhub") or None)
        _ = client._container_id
        client.close()
        return True, "Client initialized."
    except ModuleNotFoundError:
        return False, "Install library: pip install azure-eventhub"
    except Exception as e:
        return False, f"{e}"

def test_pubsub(cfg):
    try:
        from google.cloud import pubsub_v1
        creds = _google_creds_from_json(cfg.get("credentials_json"))
        publisher = pubsub_v1.PublisherClient(credentials=creds)
        project_path = f"projects/{cfg.get('project_id')}"
        list(publisher.list_topics(request={"project": project_path}).pages)
        return True, "Client initialized; topics listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install google-cloud-pubsub"
    except Exception as e:
        return False, f"{e}"

def test_kinesis(cfg):
    try:
        import boto3
        client = boto3.client("kinesis",
                              region_name=cfg.get("region_name"),
                              aws_access_key_id=cfg.get("aws_access_key_id"),
                              aws_secret_access_key=cfg.get("aws_secret_access_key"))
        client.list_streams(Limit=1)
        return True, "Client initialized; streams listable."
    except ModuleNotFoundError:
        return False, "Install library: pip install boto3"
    except Exception as e:
        return False, f"{e}"

def test_spark(cfg):
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master(cfg.get("master")).appName(cfg.get("app_name") or "ConnectorsHub").getOrCreate()
        spark.sql("SELECT 1").collect()
        spark.stop()
        return True, "Spark session started."
    except ModuleNotFoundError:
        return False, "Install library: pip install pyspark"
    except Exception as e:
        return False, f"{e}"

def test_dask(cfg):
    try:
        from distributed import Client
        c = Client(address=cfg.get("scheduler_address") or None, timeout="5s")
        c.scheduler_info()
        c.close()
        return True, "Connected to Dask scheduler."
    except ModuleNotFoundError:
        return False, "Install library: pip install dask distributed"
    except Exception as e:
        return False, f"{e}"

def test_salesforce(cfg):
    try:
        from simple_salesforce import Salesforce
        sf = Salesforce(username=cfg.get("username"), password=cfg.get("password"),
                        security_token=cfg.get("security_token"), domain=cfg.get("domain") or "login")
        _ = sf.session_id
        return True, "Authenticated."
    except ModuleNotFoundError:
        return False, "Install library: pip install simple-salesforce"
    except Exception as e:
        return False, f"{e}"

def test_servicenow(cfg):
    try:
        import pysnow
        c = pysnow.Client(instance=cfg.get("instance"), user=cfg.get("user"), password=cfg.get("password"))
        r = c.resource(api_path="/now/table/sys_user")
        _ = r.get(params={"sysparm_limit": 1})
        return True, "Authenticated; queried sys_user."
    except ModuleNotFoundError:
        return False, "Install library: pip install pysnow"
    except Exception as e:
        return False, f"{e}"

def test_jira(cfg):
    try:
        from jira import JIRA
        j = JIRA(server=cfg.get("server"), basic_auth=(cfg.get("email"), cfg.get("api_token")))
        _ = j.myself()
        return True, "Authenticated."
    except ModuleNotFoundError:
        return False, "Install library: pip install jira"
    except Exception as e:
        return False, f"{e}"

def test_sharepoint(cfg):
    try:
        import msal
        app = msal.ConfidentialClientApplication(
            client_id=cfg.get("client_id"),
            client_credential=cfg.get("client_secret"),
            authority=f"https://login.microsoftonline.com/{cfg.get('tenant_id')}",
        )
        token = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" in token:
            return True, "Token acquired for Microsoft Graph."
        return False, f"Failed to acquire token: {token.get('error_description')}"
    except ModuleNotFoundError:
        return False, "Install library: pip install msal"
    except Exception as e:
        return False, f"{e}"

def test_tableau(cfg):
    try:
        import tableauserverclient as TSC
        auth = TSC.PersonalAccessTokenAuth(token_name=cfg.get("token_name"),
                                           personal_access_token=cfg.get("token_secret"),
                                           site_id=cfg.get("site_id") or "")
        server = TSC.Server(cfg.get("server"), use_server_version=True)
        server.auth.sign_in(auth)
        server.auth.sign_out()
        return True, "Signed in with PAT."
    except ModuleNotFoundError:
        return False, "Install library: pip install tableauserverclient"
    except Exception as e:
        return False, f"{e}"

def test_gmail(cfg):
    try:
        from googleapiclient.discovery import build
        creds = _google_creds_from_json(cfg.get("credentials_json"))
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        _ = service.users().labels().list(userId="me").execute()
        return True, "Gmail API reachable; labels listed."
    except ModuleNotFoundError:
        return False, "Install libraries: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    except Exception as e:
        return False, f"{e}"

def test_msgraph(cfg):
    return test_sharepoint(cfg)

def test_weaviate(cfg):
    """
    Test Weaviate connectivity with the v4 client.
    Prefers WCS via cluster_url; otherwise uses scheme/host/port.
    Auth via API key (optional).
    """
    try:
        from urllib.parse import urlparse
        from weaviate import connect_to_wcs, connect_to_custom
        from weaviate.classes.init import Auth
    except ModuleNotFoundError:
        return False, "Install library: pip install 'weaviate-client>=4,<5'"

    client = None
    try:
        cluster_url = (cfg.get("cluster_url") or cfg.get("url") or "").strip().rstrip("/")
        api_key = (cfg.get("api_key") or "").strip() or None
        auth = Auth.api_key(api_key) if api_key else None

        if cluster_url:
            # WCS path (no timeout kw supported)
            client = connect_to_wcs(cluster_url=cluster_url, auth_credentials=auth)
        else:
            # Normalize scheme/host/port; allow users to paste a URL into Host
            scheme = (cfg.get("scheme") or "https").lower()
            host_raw = (cfg.get("host") or "").strip()
            port = cfg.get("port")

            if host_raw.startswith("http://") or host_raw.startswith("https://"):
                parsed = urlparse(host_raw)
                scheme = parsed.scheme or scheme
                host = parsed.hostname or host_raw
                port = port or parsed.port
            else:
                host = host_raw

            if not host:
                return False, "Host is required when Cluster URL is not provided."

            if port in (None, "", 0):
                port = 443 if scheme == "https" else 80

            # Self-hosted/custom path: pass host/port/secure separately
            client = connect_to_custom(
                http_host=host,
                http_port=int(port),
                http_secure=(scheme == "https"),
                auth_credentials=auth,
            )

        # Lightweight readiness/auth check
        try:
            ready = bool(getattr(client, "is_ready", lambda: True)())
        except Exception:
            # Fallback: listing collections also validates reachability+auth
            _ = list(client.collections.list_all())
            ready = True

        client.close()
        return (True, "Connected to Weaviate.") if ready else (False, "Weaviate is not ready.")
    except Exception as e:
        try:
            if client:
                client.close()
        except Exception:
            pass
        return False, f"Failed to connect to Weaviate: {e}"


    

TEST_HANDLERS = {
    "postgres": test_postgres, "mysql": test_mysql, "mssql": test_mssql, "oracle": test_oracle,
    "sqlite": test_sqlite, "trino": test_trino, "duckdb": test_duckdb, "snowflake": test_snowflake,
    "bigquery": test_bigquery, "redshift": test_redshift, "synapse": test_synapse, "mongodb": test_mongodb,
    "cassandra": test_cassandra, "redis": test_redis, "dynamodb": test_dynamodb, "neo4j": test_neo4j,
    "elasticsearch": test_elasticsearch, "cosmos": test_cosmos, "firestore": test_firestore,
    "bigtable": test_bigtable, "s3": test_s3, "azureblob": test_azureblob, "adls": test_adls, "gcs": test_gcs,
    "hdfs": test_hdfs, "kafka": test_kafka, "rabbitmq": test_rabbitmq, "eventhubs": test_eventhubs,
    "pubsub": test_pubsub, "kinesis": test_kinesis, "spark": test_spark, "dask": test_dask,
    "salesforce": test_salesforce, "servicenow": test_servicenow, "jira": test_jira, "sharepoint": test_sharepoint,
    "tableau": test_tableau, "gmail": test_gmail, "msgraph": test_msgraph, "weaviate": test_weaviate,
}

# --------------- Reusable: render Configure form into any container ---------------
def render_configure_form(container, conn: Connector):
    with container:
        # Card shell
        # st.markdown('<div class="card">', unsafe_allow_html=True)

        # Header row with inline Close (top-right) inside the card
        title_col, close_col = st.columns([10, 1])
        with title_col:
            # st.markdown('<div class="card-title-row"><h4 style="margin:0;">Configure connection profile</h4></div>',
            #             unsafe_allow_html=True)
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
            st.code(_env_snippet(conn.id, profile_name or "PROFILE", values, REG_BY_ID[conn.id].secret_keys), language="bash")

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
if st.session_state["rhs_open"]:
    with main_right:
        st.markdown('<div class="rhs-aside">', unsafe_allow_html=True)
        render_header(main_right, conn, len(REGISTRY), total_profiles_all, in_rhs=True)

        # Close button is now inside the Configure card header
        render_configure_form(main_right, conn)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with main_right:
        st.markdown("")

# ---------------------- All Configured Connections (ALWAYS in left box) ----------------------
def _run_status_check_for_all():
    _short_timeout_env()
    for cid, items in (_load_all() or {}).items():
        handler = TEST_HANDLERS.get(cid)
        for pname, cfg in (items or {}).items():
            if handler:
                try:
                    ok, msg = handler(cfg)
                except Exception as e:
                    ok, msg = False, str(e)
            else:
                ok, msg = None, "No test implemented."
            _cache_status_set(cid, pname, ok, msg)

with main_left:
    # Title inside the card
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

    # Auto-check status every render to keep it up-to-date
    _run_status_check_for_all()
    all_profiles = _load_all()

    if not all_profiles:
        st.info("You haven’t saved any connections yet.")
    else:
        # Header row (not interactive)
        h1, h2, h3, h4, h5 = st.columns([4, 3, 3, 1, 1])
        h1.markdown('<div class="rowhead">Connector</div>', unsafe_allow_html=True)
        h2.markdown('<div class="rowhead">Profile</div>', unsafe_allow_html=True)
        h3.markdown('<div class="rowhead">Status</div>', unsafe_allow_html=True)
        h4.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)
        h5.markdown('<div class="rowhead"></div>', unsafe_allow_html=True)

        # Sorted display
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

                # DELETE
                if c5.button("🗑️", key=f"del::{cid}::{pname}", help="Delete this profile"):
                    store = _load_all()
                    try:
                        # remove the profile
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
# Connections store helpers (read-only here)
# ---------------------------------------------------------------------------------
def _load_profiles_store() -> dict:
    if not CONN_STORE.exists():
        return {}
    try:
        return json.loads(CONN_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _profiles_for_connector(conn_id: str) -> List[str]:
    store = _load_profiles_store()
    return sorted(list((store.get(conn_id) or {}).keys()), key=str.lower)

def _profiles_for(connector_id: str) -> List[str]:
    """Return sorted list of saved profile names for a connector (from connections.json)."""
    store = _load_profiles_store()
    profs = store.get(connector_id, {}) or {}
    return sorted(profs.keys(), key=str.lower)

# ---------------------------------------------------------------------------------
# Pipelines store helpers
# ---------------------------------------------------------------------------------
def _pipelines_load_all() -> Dict[str, Dict[str, Any]]:
    if not PIPE_STORE.exists():
        return {}
    try:
        return json.loads(PIPE_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _pipelines_save_all(data: Dict[str, Dict[str, Any]]) -> None:
    PIPE_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _pipeline_defaults() -> Dict[str, Any]:
    return {
        "name": "",
        "source_connector": "",
        "source_profile": "",
        "destination_connector": "weaviate",
        "destination_profile": "",
        "collection": "Documents",
        # Execution knobs (placeholders for later steps)
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_docs": 50,
        "notes": "",
    }

# ---------------------------------------------------------------------------------
# Session defaults for pipeline editor
# ---------------------------------------------------------------------------------
st.session_state.setdefault("editing_pipeline_id", None)   # pipeline id (key in pipelines.json)
st.session_state.setdefault("pipeline_form_open", False)

# =================================================================================================
# UI: Pipelines Card
# =================================================================================================
with st.container(border=True):
    st.markdown("### 🧪 Pipelines")

    # Top bar
    c_a, c_b = st.columns([5, 1])
    with c_a:
        st.caption("Create, save, edit, delete and manually run pipelines that move data from a source connector to Weaviate.")
    with c_b:
        if st.button("➕ New Pipeline", use_container_width=True, key="btn_new_pipeline"):
            st.session_state["editing_pipeline_id"] = None
            st.session_state["pipeline_form_open"] = True

    pipelines = _pipelines_load_all()

    if st.session_state["pipeline_form_open"]:
        # Figure out if we're creating or editing
        is_edit = st.session_state["editing_pipeline_id"] is not None
        pid = st.session_state["editing_pipeline_id"]
        initial = pipelines.get(pid, _pipeline_defaults())
        st.subheader("Edit Pipeline" if is_edit else "Create Pipeline")

        # -------------------- SOURCE CARD (OUTSIDE FORM) --------------------
        # Hydrate stable state keys for the source pickers
        st.session_state.setdefault("pipe_src_connector", initial.get("source_connector", ""))
        st.session_state.setdefault("pipe_src_profile",   initial.get("source_profile", ""))
        st.session_state.setdefault("pipe__last_src_connector", st.session_state["pipe_src_connector"])

        with st.container(border=True):
            st.markdown("**Source**")

            profiles_store = _load_profiles_store()
            avail_sources = [cid for cid, profs in profiles_store.items() if profs]
            # Prefer non-Weaviate as source
            avail_sources = [c for c in avail_sources if c != "weaviate"] or avail_sources

            src_connector = st.selectbox(
                "Source Connector",
                options=[""] + avail_sources,
                key="pipe_src_connector",
            )

            # If connector changed, clear profile and rerun so options refresh
            if st.session_state.get("pipe__last_src_connector") != src_connector:
                st.session_state["pipe__last_src_connector"] = src_connector
                st.session_state["pipe_src_profile"] = ""
                st.rerun()

            # Populate profiles for the chosen connector
            src_profiles = _profiles_for_connector(src_connector) if src_connector else []
            st.selectbox(
                "Source Profile",
                options=[""] + src_profiles if src_profiles else [""],
                key="pipe_src_profile",
            )

            if src_connector and not src_profiles:
                st.caption(
                    "No saved profiles for this connector. Open the connector, test, then save a profile."
                )
                jump_col1, _ = st.columns([1, 3])
                if jump_col1.button("Create profile now"):
                    st.session_state["selected_id"] = src_connector
                    st.session_state["rhs_open"] = True
                    st.rerun()

        # -------------------- FORM (NAME, DESTINATION, PROCESSING, SAVE) --------------------
        with st.form("pipeline_form", clear_on_submit=False):
            # Name
            name = st.text_input(
                "Pipeline Name",
                value=initial.get("name", ""),
                placeholder="My First Pipeline",
            )

            # Destination
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

            # Processing
            with st.container(border=True):
                st.markdown("**Processing Settings**")
                c1, c2 = st.columns(2)
                with c1:
                    chunk_size = st.number_input(
                        "Chunk Size (chars)", min_value=100, step=50,
                        value=int(initial.get("chunk_size", 1000))
                    )
                    max_docs = st.number_input(
                        "Max Docs (for quick runs)", min_value=1, step=10,
                        value=int(initial.get("max_docs", 50))
                    )
                with c2:
                    chunk_overlap = st.number_input(
                        "Chunk Overlap (chars)", min_value=0, step=10,
                        value=int(initial.get("chunk_overlap", 150))
                    )
                    embedding_model = st.text_input(
                        "Embedding Model (placeholder)",
                        value=initial.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    )
                notes = st.text_area("Notes (optional)", value=initial.get("notes", ""))

            csave, ccancel = st.columns([1, 1])
            submit = csave.form_submit_button("💾 Save Pipeline", use_container_width=True)
            cancel = ccancel.form_submit_button("Cancel", use_container_width=True)

        if cancel:
            st.session_state["pipeline_form_open"] = False
            st.session_state["editing_pipeline_id"] = None
            st.rerun()

        if submit:
            # Read source selections from session state (since they’re outside the form)
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

            if errors:
                st.error("Please fix the following:\n" + "\n".join(f"- {e}" for e in errors))
            else:
                # Stable ID if editing; make a unique one if creating
                if is_edit:
                    pid_new = st.session_state["editing_pipeline_id"]
                else:
                    base = name.strip().lower().replace(" ", "_") or "pipeline"
                    pid_new, i = base, 2
                    while pid_new in pipelines:
                        pid_new = f"{base}_{i}"
                        i += 1

                pipelines[pid_new] = {
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
                }
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

            # Pretty source label
            src_meta = REG_BY_ID.get(p["source_connector"])
            src_label = f"{(src_meta.icon + ' ' + src_meta.name) if src_meta else p['source_connector']} `{p['source_profile']}`"
            dst_label = f"🧠 Weaviate `{p['destination_profile']}`"
            c2.markdown(f"{src_label} → {dst_label}")
            c3.markdown(f"`{p['collection']}`")

            # Manual Run (stub)
            if c4.button("▶️", key=f"run::{pid}", help="Run this pipeline now"):
                st.info(f"Starting pipeline **{p['name']}**… (execution wiring will be added next)")

            # Edit / Delete
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
# =================================================================================================
# =================================================================================================
# Pipeline execution wiring
#   - Resolves source/destination profiles
#   - Streams assets via ConnectorHub
#   - Parses → chunks → embeds → upserts to Weaviate
# =================================================================================================

from pathlib import Path
import tempfile
from typing import Iterable, Dict, Any, List

# If you've placed connector_hub.py next to this app:
# from connector_hub import ConnectorHub

# ---------- Helpers to resolve connection profiles ----------
def _get_profile(connector_id: str, profile_name: str) -> Dict[str, Any]:
    store = _load_profiles_store()
    profiles = store.get(connector_id, {}) or {}
    prof = profiles.get(profile_name or "", {}) or {}
    if not prof:
        raise RuntimeError(f"Profile '{profile_name}' not found for connector '{connector_id}'.")
    return prof

def _make_weaviate_from_profile(dest_profile: Dict[str, Any]):
    """
    Build a Weaviate client from a saved profile. Call ensure_collection()
    later, when you actually know the target collection name.
    """
    url = dest_profile.get("url") or dest_profile.get("cluster_url") or ""
    if not url:
        raise RuntimeError("Weaviate profile is missing 'url'.")
    api_key = dest_profile.get("api_key") or None
    tenancy = dest_profile.get("tenancy") or None
    client = make_weaviate_client(url, api_key)
    return client, tenancy

# ---------- Core runner ----------
def _run_pipeline(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single pipeline dict from pipelines.json.
    Returns a summary stats dict.
    """
    # 1) Resolve source + destination profiles (defensive)
    src_conn_id = pipeline.get("source_connector") or ""
    src_prof_id = pipeline.get("source_profile") or ""
    dst_prof_id = pipeline.get("destination_profile") or ""
    collection  = (pipeline.get("collection") or "Documents").strip()

    if not src_conn_id or not src_prof_id:
        raise RuntimeError("Pipeline is missing source connector/profile.")
    if not dst_prof_id:
        raise RuntimeError("Pipeline is missing Weaviate destination profile.")
    if not collection:
        raise RuntimeError("Collection name is empty.")

    src_prof = _get_profile(src_conn_id, src_prof_id)
    dst_prof = _get_profile("weaviate", dst_prof_id)

    # 2) Build ConnectorHub with one source (extendable later)
    cache_root = Path(tempfile.gettempdir()) / "kb_connectors_cache"
    state_root = Path(tempfile.gettempdir()) / "kb_connectors_state"
    connector_cfg: Dict[str, Any] = {"kind": src_conn_id}
    connector_cfg.update(src_prof)
    connector_cfg.setdefault("prefix", src_prof.get("prefix", ""))
    connector_cfg.setdefault("max_bytes", 50 * 1024 * 1024)

    hub = ConnectorHub(cache_root, state_root, configs=[connector_cfg])

    # 3) Destination client
    w_client, tenancy = _make_weaviate_from_profile(dst_prof)

    # 4) Execution knobs
    chunk_size    = int(pipeline.get("chunk_size", 1000))
    chunk_overlap = int(pipeline.get("chunk_overlap", 150))
    max_docs      = int(pipeline.get("max_docs", 50))

    # 5) Ensure collection exists
    ensure_collection(w_client, class_name=collection, tenancy=tenancy)

    # 6) Temps + stats
    tmp_dir = Path(tempfile.gettempdir()) / "kb_tmp_pipeline"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processed_files   = 0
    processed_chunks  = 0
    skipped_files     = 0
    errors: List[str] = []

    # 7) Stream, parse, chunk, embed, upsert
    with st.status(f"Starting pipeline **{pipeline.get('name','(unnamed)')}**", expanded=True) as status:
        try:
            total_estimate = getattr(hub, "count", None)()
        except Exception:
            total_estimate = None

        prog = st.progress(0.0) if isinstance(total_estimate, int) and total_estimate > 0 else None
        seen = 0

        for asset in hub.stream():
            if processed_files >= max_docs:
                st.write(f"⏹️ Reached max_docs={max_docs}, stopping.")
                break

            try:
                txt = guess_and_read_path(asset.name, asset.local_path, asset.size_bytes, tmp_dir).strip()
                if not txt:
                    st.write(f"⚠️ No extractable text: {asset.name}")
                    skipped_files += 1
                else:
                    chunks_raw = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    ).split_text(txt)

                    if not chunks_raw:
                        st.write(f"⚠️ No chunks produced: {asset.name}")
                        skipped_files += 1
                    else:
                        doc_id = _hash(asset.name)
                        chunks = [f"FILE: {asset.name}\n{ch}" for ch in chunks_raw]
                        vecs   = embed_texts(chunks)

                        items = []
                        for i, ch in enumerate(chunks):
                            items.append({
                                "id": _hash(f"{doc_id}:{i}"),
                                "properties": {
                                    "doc_id": doc_id,
                                    "source": asset.name,
                                    "chunk_index": i,
                                    "text": ch,
                                },
                            })

                        upserted = upsert_chunks(w_client, collection, items, vecs, tenancy=tenancy)
                        st.write(f"📄 {asset.name} → {upserted} chunks")
                        processed_chunks += upserted
                        processed_files  += 1

            except Exception as e:
                err = f"{getattr(asset, 'name', '(unknown)')}: {e}"
                errors.append(err)
                st.write(f"❌ {err}")

            seen += 1
            if prog is not None and total_estimate:
                prog.progress(min(1.0, seen / float(total_estimate)))

        status.update(
            label=f"Finished pipeline **{pipeline.get('name','(unnamed)')}**",
            state="complete"
        )

    if V4:
        try:
            w_client.close()
        except Exception:
            pass

    return {
        "files": processed_files,
        "chunks": processed_chunks,
        "skipped": skipped_files,
        "errors": errors,
    }


# ---------- Hook the “Run” buttons (keys made unique to this section) ----------
SECTION_TAG = "pipelines_exec_v1"  # change if you render another table elsewhere

def _safe_key(s: str) -> str:
    # Self-contained sanitizer (works even if `re` isn't globally imported)
    try:
        import re as _re
        return _re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s))
    except Exception:
        return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(s))

# Guard if `pipelines` not defined yet
pipelines = _pipelines_load_all() if 'pipelines' not in globals() else pipelines

for pid in sorted(pipelines.keys(), key=lambda x: pipelines[x]["name"].lower()):
    p = pipelines[pid]
    row_key = _safe_key(pid)  # stable per pipeline

    c1, c2, c3, c4, c5 = st.columns([4, 3, 2, 1, 2])

    c1.markdown(f"**{p['name']}**")

    src_meta = REG_BY_ID.get(p["source_connector"]) if 'REG_BY_ID' in globals() else None
    src_label = f"{(src_meta.icon + ' ' + src_meta.name) if src_meta else p['source_connector']} `{p['source_profile']}`"
    dst_label = f"🧠 Weaviate `{p['destination_profile']}`"

    c2.markdown(f"{src_label} → {dst_label}")
    c3.markdown(f"`{p['collection']}`")

    # Unique keys per section + pipeline
    run_key    = f"run::{SECTION_TAG}::{row_key}"
    edit_key   = f"edit_pipe::{SECTION_TAG}::{row_key}"
    delete_key = f"delete_pipe::{SECTION_TAG}::{row_key}"

    if c4.button("▶️", key=run_key, help="Run this pipeline now"):
        try:
            summary = _run_pipeline(p)
            st.success(
                f"✅ Pipeline **{p['name']}** complete — "
                f"files: {summary['files']}, chunks: {summary['chunks']}, skipped: {summary['skipped']}."
            )
            if summary["errors"]:
                with st.expander("View errors"):
                    for e in summary["errors"]:
                        st.write(f"- {e}")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    e_col, d_col = c5.columns([1, 1])
    if e_col.button("📝", key=edit_key, help="Edit"):
        st.session_state["editing_pipeline_id"] = pid
        st.session_state["pipeline_form_open"] = True
        st.rerun()

    if d_col.button("🗑️", key=delete_key, help="Delete"):
        try:
            allp = _pipelines_load_all()
            if pid in allp:
                del allp[pid]
                _pipelines_save_all(allp)
                st.success(f"Deleted pipeline **{p['name']}**.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to delete: {e}")

