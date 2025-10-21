# stores.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List, Optional

APP_DIR = Path(__file__).parent
CONN_STORE = APP_DIR / "connections.json"
PIPE_STORE = APP_DIR / "pipelines.json"

def load_connections() -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not CONN_STORE.exists(): return {}
    try: return json.loads(CONN_STORE.read_text(encoding="utf-8"))
    except Exception: return {}

def save_connections(d: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    CONN_STORE.write_text(json.dumps(d, indent=2), encoding="utf-8")

def load_pipelines() -> Dict[str, Any]:
    if not PIPE_STORE.exists(): return {}
    try: return json.loads(PIPE_STORE.read_text(encoding="utf-8"))
    except Exception: return {}

def save_pipelines(d: Dict[str, Any]) -> None:
    PIPE_STORE.write_text(json.dumps(d, indent=2), encoding="utf-8")
