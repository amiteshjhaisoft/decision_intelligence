from pathlib import Path
import json
from typing import Dict, Any

CONN_STORE = Path("connections.json")
PIPE_STORE = Path("pipelines.json")

def load_connections() -> Dict[str, Any]:
    if not CONN_STORE.exists():
        return {}
    try:
        return json.loads(CONN_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_connections(data: Dict[str, Any]) -> None:
    CONN_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_pipelines() -> Dict[str, Any]:
    if not PIPE_STORE.exists():
        return {}
    try:
        return json.loads(PIPE_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_pipelines(data: Dict[str, Any]) -> None:
    PIPE_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
