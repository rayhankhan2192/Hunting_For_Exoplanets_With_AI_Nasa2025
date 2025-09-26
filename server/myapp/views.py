import json, os, time
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pathlib import Path
from django.utils.text import slugify
import pandas as pd
from django.conf import settings

from .jobs import create_training_job, get_job, get_job_logs
from . import koiprediction

TERMINAL = {"SUCCEEDED", "FAILED"}

# === NEW: single source of truth for repo root + upload dir (shared by training & prediction) ===
REPO_ROOT = Path(__file__).resolve().parents[2]
# UPLOAD_DIR = REPO_ROOT / "DataSet" / "uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = settings.MEDIA_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Logs dir only for textual logs
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _to_int_or_none(val):
    s = (str(val).strip() if val is not None else "")
    return int(s) if s.isdigit() or (s.startswith('-') and s[1:].isdigit()) else None

@csrf_exempt
@require_http_methods(["POST"])
def start_training(request):
    """
    Start a training job (blocking until completion).
    Supports multipart (file=<csv>, satellite, model) or JSON (data_path, satellite, model).
    """

    # ---- Case A: multipart upload
    if request.FILES.get("file"):
        satellite = request.POST.get("satellite", "K2")
        model_type = request.POST.get("model", "rf")
        file_obj = request.FILES["file"]

        # only .csv allowed
        orig_name = os.path.basename(file_obj.name)
        stem, ext = os.path.splitext(orig_name)
        if ext.lower() != ".csv":
            return JsonResponse({"ok": False, "error": "Only .csv files are allowed."}, status=400)

        safe_name = f"{slugify(stem)}{ext.lower() or '.csv'}"
        dest_path = UPLOAD_DIR / safe_name

        if dest_path.exists():
            data_path = str(dest_path).replace("\\", "/")
            info = f"File exists. Using existing: {safe_name}"
        else:
            with open(dest_path, "wb+") as dest:
                for chunk in file_obj.chunks():
                    dest.write(chunk)
            data_path = str(dest_path).replace("\\", "/")
            info = f"Uploaded: {safe_name}"

        try:
            job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
        except Exception as e:
            return JsonResponse({"ok": False, "error": str(e)}, status=400)

        job_id = job_info["job_id"]

        # Block until terminal
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
            time.sleep(5)

    # ---- Case B: JSON with data_path
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


@csrf_exempt
def prediction_view(request):
    """
    POST multipart/form-data:
      - satellite: KOI (currently supported)
      - file: <csv>
      - from_csv_range (optional): start row (0-based, inclusive)
      - to_csv_range   (optional): end row   (0-based, inclusive)

    Behavior:
      - Saves the uploaded CSV into DataSet/uploads (same as training).
      - Runs prediction on full file or specified row range.
      - Saves a new CSV in the same folder that is IDENTICAL to the input
        but with ONE EXTRA COLUMN holding predictions (default 'koi_prediction').
      - If only a range is predicted, other rows get NaN in that column.
    """
    if request.method != "POST":
        return HttpResponseNotAllowed(permitted_methods=["POST"])

    try:
        # Required: satellite + file
        satellite = (request.POST.get("satellite") or "").upper().strip()
        if not satellite:
            return JsonResponse({"ok": False, "error": "Missing satellite parameter"})

        file_obj = request.FILES.get("file")
        if not file_obj:
            return JsonResponse({"ok": False, "error": "Missing CSV file"})

        # Save uploaded file in SAME folder as training
        orig_name = os.path.basename(file_obj.name)
        stem, ext = os.path.splitext(orig_name)
        if ext.lower() != ".csv":
            return JsonResponse({"ok": False, "error": "Only .csv files are allowed."}, status=400)

        safe_name = f"{slugify(stem)}{ext.lower() or '.csv'}"
        tmp_path = UPLOAD_DIR / safe_name
        with open(tmp_path, "wb+") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Accept both correct key and the earlier typo just in case
        from_raw = request.POST.get("from_csv_range")
        if from_raw is None:
            from_raw = request.POST.get("from_csv_rabge") 
        to_raw = request.POST.get("to_csv_range")

        from_row = _to_int_or_none(from_raw)
        to_row   = _to_int_or_none(to_raw)

        # Load once if needed
        df_len = None
        if from_row is None and to_row is None:
            # No range provided -> predict full CSV
            row_numbers = None
        else:
            if from_row is None:
                from_row = 0
            if to_row is None:
                if df_len is None:
                    df_len = pd.read_csv(tmp_path, comment="#").shape[0]
                to_row = df_len - 1

            if df_len is None:
                df_len = pd.read_csv(tmp_path, comment="#").shape[0]

            from_row = max(0, from_row)
            to_row   = min(df_len - 1, to_row)

            if from_row > to_row:
                return JsonResponse({"ok": False, "error": "Invalid row range: from_row > to_row"})

            row_numbers = list(range(from_row, to_row + 1))

        # Dispatch prediction
        if satellite == "KOI":
            from . import koiprediction
            results_df = koiprediction.predict_from_csv(str(tmp_path), row_numbers)
        else:
            return JsonResponse({"ok": False, "error": f"Satellite {satellite} not supported yet"})

        if results_df is None or results_df.empty:
            return JsonResponse({"ok": False, "error": "Prediction failed"})

        # Merge predictions into the ORIGINAL CSV (append columns)
        df_raw = pd.read_csv(tmp_path, comment="#")

        # Identify prediction-related columns to append
        pred_cols = []
        # Always include Predicted_Class if present
        if "Predicted_Class" in results_df.columns:
            pred_cols.append("Predicted_Class")
        # Common extras
        for c in ["Confidence", "Match"]:
            if c in results_df.columns and c not in pred_cols:
                pred_cols.append(c)
        # All probability columns (Prob_*)
        prob_cols = [c for c in results_df.columns if str(c).startswith("Prob_")]
        pred_cols.extend([c for c in prob_cols if c not in pred_cols])

        # If nothing matched, fall back to first column as class and try to find probs heuristically
        if not pred_cols:
            # class-like first column
            first_col = results_df.columns[0]
            pred_cols.append(first_col)
            # any columns containing probability info
            pred_cols.extend([c for c in results_df.columns if "prob" in str(c).lower() and c != first_col])

        # Ensure target columns exist in output and init with NaN
        for c in pred_cols:
            out_col = c
            # avoid collision with original CSV columns
            if out_col in df_raw.columns:
                i = 2
                while f"{c}__pred{i}" in df_raw.columns:
                    i += 1
                out_col = f"{c}__pred{i}"
            df_raw[out_col] = pd.NA

        # Make a mapping from results col -> actual output col name used
        out_name_map = {}
        for c in pred_cols:
            if c in df_raw.columns:
                # if we didn't rename, it means no collision
                out_name_map[c] = c
            else:
                # find the created one (either exact c or c__pred*)
                if c in df_raw.columns:
                    out_name_map[c] = c
                else:
                    # search created variant
                    found = None
                    if c in pred_cols:
                        # try exact first
                        if c in df_raw.columns:
                            found = c
                        else:
                            # find matching prefix
                            for cc in df_raw.columns:
                                if cc == c or cc.startswith(f"{c}__pred"):
                                    found = cc
                                    break
                    out_name_map[c] = found or c  # fallback

        # Fill values
        if row_numbers is None:
            # Full file
            if len(results_df) != len(df_raw):
                return JsonResponse({"ok": False, "error": "Prediction length mismatch with input rows"}, status=500)
            for c in pred_cols:
                df_raw[out_name_map[c]] = results_df[c].values if c in results_df.columns else pd.NA
            from_resp, to_resp = 0, len(df_raw) - 1
        else:
            # Range only
            if len(results_df) != len(row_numbers):
                return JsonResponse({"ok": False, "error": "Prediction length mismatch with requested row range"}, status=500)
            for c in pred_cols:
                vals = results_df[c].values if c in results_df.columns else [pd.NA] * len(row_numbers)
                df_raw.loc[row_numbers, out_name_map[c]] = list(vals)
            from_resp, to_resp = from_row, to_row

        # Save predicted CSV in SAME folder as training (unique name to avoid locks)
        ts = time.strftime("%Y%m%d-%H%M%S")
        pred_name = f"{Path(safe_name).stem}__pred_{satellite}_{ts}.csv"
        output_file = (UPLOAD_DIR / pred_name)
        df_raw.to_csv(output_file, index=False)
        csv_url = f"{settings.MEDIA_URL}uploads/{pred_name}"

        # Respond (JSON FORMAT UNCHANGED)
        return JsonResponse({
            "ok": True,
            "satellite": satellite,
            "from_row": from_resp if row_numbers is not None else 0,
            "to_row": to_resp if row_numbers is not None else (df_len if df_len is not None else pd.read_csv(tmp_path, comment="#").shape[0]) - 1,
            "total_predicted": len(results_df),
            "csv_file": request.build_absolute_uri(csv_url), 
            "results": results_df.to_dict(orient="records"),
        })

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)})