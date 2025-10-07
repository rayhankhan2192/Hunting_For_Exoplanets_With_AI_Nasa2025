import uuid, threading, traceback, time, os, json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from django.conf import settings
from main.train import main as train_main
from main.train_all import main as trail_all_main  

JOBS = {}

MEDIA_ROOT = Path(settings.MEDIA_ROOT).resolve()
LOG_DIR = (MEDIA_ROOT / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
VALID_SATELLITES = {"K2", "TOI", "KOI"}
VALID_MODELS = {"rf", "gb", "xgb", "dt", "lr", "svm"}  

def _status_path(job_id: str) -> Path:
    return LOG_DIR / f"{job_id}.status.json"

def _log_path(job_id: str) -> Path:
    return LOG_DIR / f"{job_id}.log"

def _write_status(job_id: str, payload: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_status_path(job_id), "w", encoding="utf-8") as f:
        json.dump(payload, f)

def _read_status(job_id: str) -> dict | None:
    p = _status_path(job_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _update_status(job_id: str, **fields):
    state = JOBS.get(job_id) or _read_status(job_id) or {}
    state.update(fields)
    JOBS[job_id] = state
    _write_status(job_id, state)

def _run_training(job_id: str, data_path: str, satellite: str, model_type: str | None, is_trainall: bool):
    log_file = _log_path(job_id)
    start = time.time()
    _update_status(
        job_id,
        status="RUNNING",
        started_at=start,
        params={"data_path": data_path, "satellite": satellite, "model": model_type, "is_trainall": is_trainall},
        result=None,
        error=None,
        progress=0,
        log_path=str(log_file).replace("\\", "/"),
    )

    with open(log_file, "w", encoding="utf-8") as lf:
        with redirect_stdout(lf), redirect_stderr(lf):
            print(f"[JOB {job_id}] Starting training...")
            print(f"Params: data_path={data_path}, satellite={satellite}, model={model_type}, is_trainall={is_trainall}")
            print(f"CWD: {os.getcwd()}")

            try:
                _update_status(job_id, progress=5)

                #choose path
                if is_trainall:
                    print(f"[JOB {job_id}] Running train-ALL pipeline...")
                    result = trail_all_main(data_path=data_path, 
                                            satellite=satellite, 
                                            is_trainall=True)
                else:
                    print(f"[JOB {job_id}] Running single-model pipeline: {model_type}")
                    result = train_main(data_path=data_path, 
                                        satellite=satellite, 
                                        model_type=model_type)

                _update_status(job_id, progress=80)
                elapsed = time.time() - start

                if isinstance(result, dict):
                    metrics = dict(result) 
                elif isinstance(result, tuple) and len(result) >= 6:
                    accuracy, cv_mean, cv_std, auc_score, cm_img_path, model_path = result[:6]
                    metrics = {
                        "accuracy": float(accuracy),
                        "cv_mean": float(cv_mean),
                        "cv_std": float(cv_std),
                        "auc_score": float(auc_score),
                        "cm_image_path": str(cm_img_path),
                        "cm_image_url": None,
                        "cm_norm_image_path": None,
                        "cm_norm_image_url": None,
                        "model_path": str(model_path),
                        "model_url": None,
                    }
                elif isinstance(result, str):
                    metrics = {
                        "accuracy": None,
                        "cv_mean": None,
                        "cv_std": None,
                        "auc_score": None,
                        "cm_image_path": None,
                        "cm_image_url": None,
                        "cm_norm_image_path": None,
                        "cm_norm_image_url": None,
                        "model_path": result,
                        "model_url": None,
                    }
                else:
                    metrics = {"raw_result": result}

                metrics["elapsed_sec"] = elapsed
                print(f"[JOB {job_id}] Training finished in {elapsed:.2f}s")
                print(f"[JOB {job_id}] Result summary: { {k: metrics[k] for k in list(metrics)[:8]} }")
                _update_status(
                    job_id,
                    status="SUCCEEDED",
                    finished_at=time.time(),
                    progress=100,
                    result=metrics,
                    error=None,
                )
            except Exception as e:
                traceback.print_exc()
                _update_status(
                    job_id,
                    status="FAILED",
                    finished_at=time.time(),
                    progress=100,
                    result=None,
                    error=str(e),
                )

def create_training_job(data_path: str, satellite: str, model_type: str | None = None, is_trainall: bool = False) -> dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if satellite not in VALID_SATELLITES:
        raise ValueError(f"Invalid satellite '{satellite}'. Allowed: {sorted(VALID_SATELLITES)}")

    # Only validate model when NOT training all
    if not is_trainall:
        if not model_type:
            raise ValueError("model_type is required when is_trainall=False")
        if model_type not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model_type}'. Allowed: {sorted(VALID_MODELS)}")

    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "status": "PENDING",
        "created_at": time.time(),
        "params": {"data_path": data_path, "satellite": satellite, "model": model_type, "is_trainall": is_trainall},
        "result": None,
        "error": None,
        "log_path": str(_log_path(job_id)).replace("\\", "/"),
        "progress": 0,
    }
    JOBS[job_id] = payload
    _write_status(job_id, payload)
    t = threading.Thread(
        target=_run_training,
        args=(job_id, data_path, satellite, model_type, is_trainall),
        daemon=True
    )
    t.start()
    return {"job_id": job_id, "status": "PENDING"}

def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id) or _read_status(job_id)

def get_job_logs(job_id: str, tail: int | None = None) -> str:
    p = _log_path(job_id)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    if tail and tail > 0:
        return "\n".join(text.splitlines()[-tail:])
    return text
