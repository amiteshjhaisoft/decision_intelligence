# runner.py
# runner.py
from typing import Optional, Any
from pipelines_app import load_pipelines, run_pipeline  # from the canvas code (pipelines_app.py)

def run_pipeline_by_id(pipeline_id: str, uploads: Optional[list[Any]] = None, ui_log=None):
    store = load_pipelines()
    if pipeline_id not in store:
        raise KeyError(f"Pipeline id not found: {pipeline_id}")
    return run_pipeline(store[pipeline_id], uploads=uploads, ui_log=ui_log)
