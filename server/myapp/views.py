import json, os, time
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path
from django.utils.text import slugify

from .jobs import create_training_job, get_job, get_job_logs

TERMINAL = {"SUCCEEDED", "FAILED"}

@csrf_exempt
@require_http_methods(["POST"])
def start_training(request):
    """

    Start a training job.

    - Multipart/form-data with file:
        file=<csv>, satellite=K2, model=rf
      Saves under <REPO_ROOT>/datasets/uploads/
      If a file with the same name already exists, it is NOT overwritten;
      we just use the existing file path for training.

    - JSON with data_path (legacy)


    Start training and BLOCK until it finishes (no timeout).
    Supports:
      - form-data: file=<csv>, satellite, model
      - JSON: {"data_path": "...", "satellite": "...", "model": "..."}
    """

    #Case A: multipart upload
    if request.FILES.get("file"):
        satellite = request.POST.get("satellite", "K2")
        model_type = request.POST.get("model", "rf")
        file_obj = request.FILES["file"]

        # only .csv allowed
        orig_name = os.path.basename(file_obj.name)
        stem, ext = os.path.splitext(orig_name)
        if ext.lower() != ".csv":
            return JsonResponse({"ok": False, "error": "Only .csv files are allowed."}, status=400)

        # repo root: .../Hunting_For_Exoplanets_With_AI_Nasa2025/
        REPO_ROOT = Path(__file__).resolve().parents[2]
        UPLOAD_DIR = REPO_ROOT / "DataSet" / "uploads"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = f"{slugify(stem)}{ext.lower() or '.csv'}"
        dest_path = UPLOAD_DIR / safe_name

        if dest_path.exists():
            data_path = str(dest_path)
            info = f"File exists. Using existing: {safe_name}"
        else:
            with open(dest_path, "wb+") as dest:
                for chunk in file_obj.chunks():
                    dest.write(chunk)
            data_path = str(dest_path)
            info = f"Uploaded: {safe_name}"

        data_path = data_path.replace("\\", "/")  # normalize

        try:
            job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
        except Exception as e:
            return JsonResponse({"ok": False, "error": str(e)}, status=400)

        job_id = job_info["job_id"]

        # Block until terminal state
        while True:
            state = get_job(job_id)
            if state and state.get("status") in TERMINAL:
                if state["status"] == "SUCCEEDED":
                    return JsonResponse({
                        "ok": True,
                        "message": "Training is successful",
                        "job_id": job_id,
                        "status": state["status"],
                        "result": state.get("result"),
                        "info": info
                    }, status=200)
                else:
                    return JsonResponse({
                        "ok": False,
                        "message": "Training failed",
                        "job_id": job_id,
                        "status": state["status"],
                        "error": state.get("error"),
                        "info": info
                    }, status=500)
            time.sleep(5)  # poll every 5s

    # Case B: JSON with data_path
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON body")

    data_path = payload.get("data_path")
    satellite = payload.get("satellite", "K2")
    model_type = payload.get("model", "rf")

    if not data_path:
        return HttpResponseBadRequest("Missing 'data_path' or 'file'")

    try:
        job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    job_id = job_info["job_id"]

    # Block until terminal state
    while True:
        state = get_job(job_id)
        if state and state.get("status") in TERMINAL:
            if state["status"] == "SUCCEEDED":
                return JsonResponse({
                    "ok": True,
                    "message": "Training is successful",
                    "job_id": job_id,
                    "status": state["status"],
                    "result": state.get("result"),
                }, status=200)
            else:
                return JsonResponse({
                    "ok": False,
                    "message": "Training failed",
                    "job_id": job_id,
                    "status": state["status"],
                    "error": state.get("error"),
                }, status=500)
        time.sleep(5)


@require_http_methods(["GET"])
def training_status(request, job_id: str):
    job = get_job(job_id)
    if not job:
        return HttpResponseNotFound("Job not found")
    return JsonResponse({
        "ok": True,
        "job_id": job_id,
        "status": job["status"],
        "params": job["params"],
        "result": job["result"],
        "error": job["error"],
    })


@require_http_methods(["GET"])
def training_logs(request, job_id: str):
    job = get_job(job_id)
    if not job:
        return HttpResponseNotFound("Job not found")
    tail_param = request.GET.get("tail")
    try:
        tail = int(tail_param) if tail_param is not None else None
    except ValueError:
        tail = None
    logs = get_job_logs(job_id, tail=tail)
    return JsonResponse({"ok": True, "job_id": job_id, "tail": tail, "logs": logs})