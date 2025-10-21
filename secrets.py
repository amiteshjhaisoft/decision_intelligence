# Author: Amitesh Jha | iSOFT

# secrets.py
from __future__ import annotations
from typing import Any, Dict, Union, List
import os

try:
    import keyring  # uses Windows Credential Manager, macOS Keychain, or Secret Service on Linux
except Exception:
    keyring = None

def _from_env(var: str) -> str:
    val = os.getenv(var)
    if val is None:
        raise KeyError(f"Missing environment variable: {var}")
    return val

def _from_keyring(service: str, name: str) -> str:
    if keyring is None:
        raise RuntimeError("keyring not installed or unavailable.")
    val = keyring.get_password(service, name)
    if not val:
        raise KeyError(f"Secret not found in keyring: service='{service}', name='{name}'")
    return val

def resolve_secrets(obj: Any) -> Any:
    """
    Recursively resolve placeholders:
      - {"$env": "VAR_NAME"}                   -> value from environment
      - {"$secret": "service/name"}            -> value from OS keyring
      - {"$secret": ["service","name"]}        -> value from OS keyring
    Everything else is returned as-is.
    """
    if isinstance(obj, dict):
        if "$env" in obj and isinstance(obj["$env"], str):
            return _from_env(obj["$env"])
        if "$secret" in obj:
            spec = obj["$secret"]
            if isinstance(spec, str):
                if "/" not in spec:
                    raise ValueError("For $secret, use 'service/name' or ['service','name'].")
                service, name = spec.split("/", 1)
                return _from_keyring(service, name)
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                return _from_keyring(spec[0], spec[1])
        # Recurse
        return {k: resolve_secrets(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [resolve_secrets(v) for v in obj]

    return obj  # primitive