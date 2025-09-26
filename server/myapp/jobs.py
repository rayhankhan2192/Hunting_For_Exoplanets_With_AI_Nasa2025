import uuid
import threading
import traceback
import time
import os
import io
from contextlib import redirect_stdout, redirect_stderr

from main.train import main as train_main 

JOBS = {}

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

VALID_SATELLITES = {"K2", "TOI", "KOI"}  # keep in sync with your code path
VALID_MODELS = {"rf", "xgb", "dt", "grdb", "logreg", "svm"}

def _run_training(job_id: str, data_path: str, satellite: str, model_type: str):
    """Background runner that executes training and captures logs to file."""
    log_path = JOBS[job_id]["log_path"]
    JOBS[job_id]["status"] = "RUNNING"

    # Capture stdout/stderr into a log file so you can tail via endpoint
    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"[JOB {job_id}] Starting training...")
            print(f"Params: data_path={data_path}, satellite={satellite}, model_type={model_type}")
            print(f"Working dir: {os.getcwd()}")
            start = time.time()
            try:
                # Call your existing training main
                train_main(data_path=data_path, satellite=satellite, model_type=model_type)

                result_path = train_main(data_path=data_path, satellite=satellite, model_type=model_type)
                elapsed = time.time() - start
                JOBS[job_id]["status"] = "SUCCEEDED"
                JOBS[job_id]["result"] = {
                    "elapsed_sec": elapsed,
                    "model_path": result_path,
}
            except Exception as e:
                print(f"[JOB {job_id}] ERROR: {e}")
                traceback.print_exc()
                JOBS[job_id]["status"] = "FAILED"
                JOBS[job_id]["error"] = str(e)


def create_training_job(data_path: str, satellite: str, model_type: str) -> dict:
    """Validate inputs, register job, spawn thread, return job metadata."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if satellite not in VALID_SATELLITES:
        raise ValueError(f"Invalid satellite '{satellite}'. Allowed: {sorted(VALID_SATELLITES)}")

    if model_type not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model_type}'. Allowed: {sorted(VALID_MODELS)}")

    job_id = str(uuid.uuid4())
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")

    JOBS[job_id] = {
        "status": "PENDING",
        "created_at": time.time(),
        "params": {"data_path": data_path, "satellite": satellite, "model_type": model_type},
        "log_path": log_path,
        "result": None,
        "error": None,
    }

    t = threading.Thread(
        target=_run_training,
        args=(job_id, data_path, satellite, model_type),
        daemon=True
    )
    t.start()

    return {"job_id": job_id, "status": "PENDING"}


def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)


def get_job_logs(job_id: str, tail: int | None = None) -> str:
    job = JOBS.get(job_id)
    if not job:
        return ""
    path = job.get("log_path")
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if tail and tail > 0:
        # Return last N lines
        lines = content.splitlines()
        return "\n".join(lines[-tail:])
    return content