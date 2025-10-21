from pathlib import Path
from typing import Iterable, Dict, Any
from plugins.registry import register_source

@register_source("localfs")
class LocalFolderSource:
    def __init__(self, cfg: Dict[str, Any]):
        self.folder = cfg.get("folder", "./KB")

    def rows(self, max_docs: int = 0) -> Iterable[Dict[str, Any]]:
        count = 0
        for p in Path(self.folder).rglob("*.txt"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            yield {"text": txt, "source_path": str(p)}
            count += 1
            if max_docs and count >= max_docs:
                break
