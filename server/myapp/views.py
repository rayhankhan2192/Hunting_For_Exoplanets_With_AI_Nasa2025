import json, os, time
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path
from django.utils.text import slugify

from .jobs import create_training_job, get_job, get_job_logs
from . import koiprediction

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


import pandas as pd
# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
@csrf_exempt
def prediction_view(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Only POST allowed"})

    try:
        # === Required params ===
        satellite = request.POST.get("satellite", "").upper().strip()
        if not satellite:
            return JsonResponse({"ok": False, "error": "Missing satellite parameter"})

        file_obj = request.FILES.get("file")
        if not file_obj:
            return JsonResponse({"ok": False, "error": "Missing CSV file"})

        # Save uploaded file temporarily
        tmp_path = LOG_DIR / file_obj.name
        with open(tmp_path, "wb+") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # === Optional row range ===
        try:
            from_row = int(request.POST.get("from_csv_range", 0))
            to_row = request.POST.get("to_csv_range")
            to_row = int(to_row) if to_row is not None else -1
        except ValueError:
            return JsonResponse({"ok": False, "error": "Invalid row range"})

        # If no to_row → full CSV
        if to_row == -1:
            df = pd.read_csv(tmp_path, comment="#")
            to_row = len(df) - 1

        row_numbers = list(range(from_row, to_row + 1))

        # === Dispatch prediction ===
        if satellite == "KOI":
            results_df = koiprediction.predict_from_csv(str(tmp_path), row_numbers)
        else:
            return JsonResponse({"ok": False, "error": f"Satellite {satellite} not supported yet"})

        if results_df is None or results_df.empty:
            return JsonResponse({"ok": False, "error": "Prediction failed"})

        # === Save predicted CSV ===
        output_file = LOG_DIR / f"predicted_{satellite}.csv"
        results_df.to_csv(output_file, index=False)

        # === Build response ===
        return JsonResponse({
            "ok": True,
            "satellite": satellite,
            "from_row": from_row,
            "to_row": to_row,
            "total_predicted": len(results_df),
            "csv_file": str(output_file),
            "results": results_df.to_dict(orient="records"),
        })

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)})