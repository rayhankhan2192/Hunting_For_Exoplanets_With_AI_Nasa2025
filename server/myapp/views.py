import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.core import signing
from django.http import HttpResponseBadRequest, HttpResponseNotFound
from django.utils.text import slugify
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


from .jobs import create_training_job, get_job, get_job_logs
from .mergecsv import merge_csvs
from . import koiprediction


TERMINAL = {"SUCCEEDED", "FAILED"}

MEDIA_ROOT: Path = Path(settings.MEDIA_ROOT).resolve()
UPLOAD_DIR: Path = (MEDIA_ROOT / "uploads").resolve()
LOG_DIR: Path = (MEDIA_ROOT / "logs").resolve()
MERGE_DIR: Path = (MEDIA_ROOT / "mergefiles").resolve() 

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MERGE_DIR.mkdir(parents=True, exist_ok=True)

_SIGNER = signing.TimestampSigner(salt="koi-merge") 

def _to_int_or_none(val) -> Optional[int]:
    s = (str(val).strip() if val is not None else "")
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return None


def _abs_media_url(request, rel_path: str) -> str:
    """Build an absolute URL for a media-relative path."""
    rel = rel_path.replace("\\", "/").lstrip("/")
    return request.build_absolute_uri(f"/media/{rel}")


def _make_merge_token(rel_media_path: str) -> str:
    """
    Create a signed token that encodes the relative media path (e.g., 'uploads/merged__...csv').
    You can require this token in a training endpoint to enforce 'merge first, then train'.
    """
    payload = json.dumps({"rel": rel_media_path})
    return _SIGNER.sign(payload)


def _save_upload_to_media(file, suggested_name: Optional[str] = None) -> str:
    """
    Save an uploaded file into MEDIA_ROOT/uploads and return the stored filename.
    """
    if not file:
        raise ValueError("No file object provided")

    name = suggested_name or getattr(file, "name", None) or "upload.csv"
    stem = slugify(Path(name).stem) or "upload"
    filename = f"{stem}.csv"

    out_path = (UPLOAD_DIR / filename)
    i = 1
    while out_path.exists():
        filename = f"{stem}-{i}.csv"
        out_path = (UPLOAD_DIR / filename)
        i += 1
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
        out_path = (UPLOAD_DIR / filename)

    with open(out_path, "wb") as f:
        chunks = file.chunks() if hasattr(file, "chunks") else [file.read()]
        for chunk in chunks:
            f.write(chunk)

    return filename 

class StartTrainingView(APIView):
    """
    Start a training job.
    - Multipart: upload a CSV via `file`, with optional `satellite`, `model`.
    - JSON: pass an existing `data_path` under MEDIA_ROOT, with optional `satellite`, `model`.
    """

    def post(self, request):
        def _json_ok(job_id, info=None):
            status_url = request.build_absolute_uri(f"/api/train/{job_id}/status")
            logs_url = request.build_absolute_uri(f"/api/train/{job_id}/logs")
            return Response(
                {
                    "ok": True,
                    "message": "Training started",
                    "job_id": job_id,
                    "status": "PENDING",
                    "status_url": status_url,
                    "logs_url": logs_url,
                    "info": info or "",
                },
                status=status.HTTP_202_ACCEPTED,
            )
        if request.FILES.get("file"):
            satellite = (request.POST.get("satellite") or "K2").strip()
            model_type = (request.POST.get("model") or "rf").strip()
            file = request.FILES["file"]

            orig_name = os.path.basename(file.name)
            stem, ext = os.path.splitext(orig_name)
            if ext.lower() != ".csv":
                return Response({"ok": False, "error": "Only .csv files allowed"}, status=400)

            safe_name = f"{slugify(stem)}{ext.lower() or '.csv'}"
            dest_path = UPLOAD_DIR / safe_name

            if dest_path.exists():
                data_path = str(dest_path).replace("\\", "/")
                info = f"File exists. Using existing: {safe_name}"
            else:
                with open(dest_path, "wb+") as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)
                data_path = str(dest_path).replace("\\", "/")
                info = f"Uploaded: {safe_name}"

            try:
                job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
            except Exception as e:
                return Response({"ok": False, "error": str(e)}, status=400)
            return _json_ok(job_info["job_id"], info=info)
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            return HttpResponseBadRequest("Invalid JSON body")
        data_path = (payload.get("data_path") or "").strip()
        satellite = (payload.get("satellite") or "K2").strip()
        model_type = (payload.get("model") or "rf").strip()

        if not data_path:
            return HttpResponseBadRequest("Missing 'data_path' or 'file'")
        try:
            job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
        except Exception as e:
            return Response({"ok": False, "error": str(e)}, status=400)

        return _json_ok(job_info["job_id"])

from rest_framework.parsers import MultiPartParser, FormParser
def _to_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default
from django.urls import reverse
class TrainAllView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"detail": "Provide a CSV file in 'file' field."}, status=400)
        satellite = (request.data.get("satellite") or "KOI").strip().upper()
        is_trainall = _to_bool(request.data.get("is_trainall"), default=False)
        try:
            stored_name = _save_upload_to_media(file_obj) 
        except Exception as e:
            return Response({"detail": f"Upload failed: {e}"}, status=400)

        data_path = str((UPLOAD_DIR / stored_name).resolve())
        file_url = _abs_media_url(request, f"uploads/{stored_name}")
        try:
            job_id = None
            try:
                job_info = create_training_job(
                    data_path=data_path, 
                    satellite=satellite, 
                    model_type=None,
                    is_trainall=is_trainall
                )
                job_id = job_info.get("job_id")  # <-- IMPORTANT
                if not job_id:
                    return Response({"detail": "Failed to create job."}, status=500)
            except Exception as e:
                return Response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            return Response({"detail": f"Job creation failed: {e}"}, status=500)

        poll_url = request.build_absolute_uri(f"{request.path}?job_id={job_id}")
        logs_url = request.build_absolute_uri(f"{request.path}?job_id={job_id}&tail=800")

        return Response(
            {
                "job_id": job_id,
                "status": "QUEUED",
                "poll_url": poll_url,
                "logs_url": logs_url,
                "file_url": file_url,
                "satellite": satellite,
                "is_trainall": is_trainall,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    def get(self, request):
        job_id = request.query_params.get("job_id")
        if not job_id:
            return Response({"detail": "job_id is required."}, status=400)

        tail = _to_int_or_none(request.query_params.get("tail")) or 400
        tail = max(50, min(tail, 5000))

        job = get_job(job_id)
        if not job:
            return Response({"detail": "Job not found."}, status=404)

        status_str = job.get("status", "UNKNOWN")
        progress = job.get("progress", 0)
        started_at = job.get("started_at")
        updated_at = job.get("updated_at")
        result = job.get("result")
        try:
            logs_tail = get_job_logs(job_id, tail=tail)
        except TypeError:
            try:
                logs_tail = get_job_logs(job_id, tail)
            except Exception:
                logs_tail = ""
        if result and isinstance(result, dict):
            for key in ("cm_image_url", "cm_norm_image_url", "model_url"):
                val = result.get(key)
                if isinstance(val, str) and val.startswith("/media/"):
                    result[key] = request.build_absolute_uri(val)

            for path_key, url_key in [
                ("cm_image_path", "cm_image_url"),
                ("cm_norm_image_path", "cm_norm_image_url"),
                ("model_path", "model_url"),
            ]:
                if result.get(url_key) is None and result.get(path_key):
                    try:
                        rel = str(Path(result[path_key]).resolve()).replace(str(MEDIA_ROOT), "").lstrip("\\/").replace("\\", "/")
                        result[url_key] = _abs_media_url(request, rel)
                    except Exception:
                        pass
        return Response(
            {
                "job_id": job_id,
                "status": status_str,
                "progress": progress,
                "started_at": started_at,
                "updated_at": updated_at,
                "logs_tail": logs_tail,
                "result": result if status_str in TERMINAL else None,
            },
            status=200,
        )

class TrainingStatusView(APIView):
    def get(self, request, job_id):
        job = get_job(job_id)
        if not job:
            return HttpResponseNotFound("Job not found")
        return Response(
            {
                "ok": True,
                "job_id": job_id,
                "status": job["status"],
                "params": job["params"],
                "result": job["result"],
                "error": job["error"],
            }
        )


class TrainingLogsView(APIView):
    def get(self, request, job_id):
        job = get_job(job_id)
        if not job:
            return HttpResponseNotFound("Job not found")
        tail_param = request.GET.get("tail")
        try:
            tail = int(tail_param) if tail_param is not None else None
        except ValueError:
            tail = None
        logs = get_job_logs(job_id, tail=tail)
        return Response({"ok": True, "job_id": job_id, "tail": tail, "logs": logs})


from . import k2prediction
class PredictionView(APIView):
    """
    POST (multipart form):
      - satellite: "KOI" | "K2" | "TESS" | "TOI" (case-insensitive)
      - model_type: "xgb" | "rf" | "logreg" | "svc" | ... (optional, default "xgb")
      - file: CSV
      - from_csv_range: optional int
      - to_csv_range: optional int
    """

    def post(self, request, *args, **kwargs):
        try:
            satellite = (request.data.get("satellite") or "").upper().strip()
            if not satellite:
                return Response({"ok": False, "error": "Missing satellite parameter"},
                                status=status.HTTP_400_BAD_REQUEST)

            model_type = (request.data.get("model_type") or "xgb").lower().strip()

            file = request.FILES.get("file")
            if not file:
                return Response({"ok": False, "error": "Missing CSV file"},
                                status=status.HTTP_400_BAD_REQUEST)

            orig_name = os.path.basename(file.name)
            stem, ext = os.path.splitext(orig_name)
            if ext.lower() != ".csv":
                return Response({"ok": False, "error": "Only .csv files are allowed."},
                                status=status.HTTP_400_BAD_REQUEST)

            safe_name = f"{slugify(stem)}{ext.lower() or '.csv'}"
            tmp_path = UPLOAD_DIR / safe_name
            with open(tmp_path, "wb+") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # range
            from_raw = request.data.get("from_csv_range") or request.data.get("from_csv_rabge")
            to_raw = request.data.get("to_csv_range")
            from_row = _to_int_or_none(from_raw)
            to_row = _to_int_or_none(to_raw)

            df_len = None
            if from_row is None and to_row is None:
                row_numbers = None
            else:
                if df_len is None:
                    df_len = pd.read_csv(tmp_path, comment="#").shape[0]
                if from_row is None:
                    from_row = 0
                if to_row is None:
                    to_row = df_len - 1

                from_row = max(0, from_row)
                to_row = min(df_len - 1, to_row)
                if from_row > to_row:
                    return Response({"ok": False, "error": "Invalid row range: from_row > to_row"},
                                    status=status.HTTP_400_BAD_REQUEST)
                row_numbers = list(range(from_row, to_row + 1))

            # dispatch by satellite
            if satellite == "KOI":
                results_df = koiprediction.predict_from_csv(
                    str(tmp_path), satellite=satellite, row_numbers=row_numbers, model_type=model_type
                )
            elif satellite == "K2":
                results_df = k2prediction.predict_from_csv(
                    str(tmp_path), satellite=satellite, row_numbers=row_numbers, model_type=model_type
                )
            else:
                return Response({"ok": False, "error": f"Satellite {satellite} not supported yet"},
                                status=status.HTTP_400_BAD_REQUEST)


            if results_df is None or results_df.empty:
                return Response({"ok": False, "error": "Prediction failed"},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            df_raw = pd.read_csv(tmp_path, comment="#")

            # prediction/proba columns to append
            pred_cols = []
            if "Predicted_Class" in results_df.columns:
                pred_cols.append("Predicted_Class")
            for c in ["Confidence", "Match"]:
                if c in results_df.columns and c not in pred_cols:
                    pred_cols.append(c)
            prob_cols = [c for c in results_df.columns if str(c).startswith("Prob_")]
            pred_cols.extend([c for c in prob_cols if c not in pred_cols])

            if not pred_cols:
                first_col = results_df.columns[0]
                pred_cols.append(first_col)
                pred_cols.extend([c for c in results_df.columns if "prob" in str(c).lower() and c != first_col])

            # ensure no collisions
            for c in pred_cols:
                out_col = c
                if out_col in df_raw.columns:
                    i = 2
                    while f"{c}__pred{i}" in df_raw.columns:
                        i += 1
                    out_col = f"{c}__pred{i}"
                df_raw[out_col] = pd.NA

            # map results -> df_raw output col
            out_name_map = {}
            for c in pred_cols:
                if c in df_raw.columns:
                    out_name_map[c] = c
                else:
                    match = next((cc for cc in df_raw.columns if cc == c or cc.startswith(f"{c}__pred")), None)
                    out_name_map[c] = match or c

            # write back rows
            if row_numbers is None:
                if len(results_df) != len(df_raw):
                    return Response({"ok": False, "error": "Prediction length mismatch with input rows"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                for c in pred_cols:
                    df_raw[out_name_map[c]] = results_df[c].values if c in results_df.columns else pd.NA
                from_resp, to_resp = 0, len(df_raw) - 1
            else:
                if len(results_df) != len(row_numbers):
                    return Response({"ok": False, "error": "Prediction length mismatch with requested row range"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                for c in pred_cols:
                    vals = results_df[c].values if c in results_df.columns else [pd.NA] * len(row_numbers)
                    df_raw.loc[row_numbers, out_name_map[c]] = list(vals)
                from_resp, to_resp = row_numbers[0], row_numbers[-1]

            ts = time.strftime("%Y%m%d-%H%M%S")
            pred_name = f"{Path(safe_name).stem}__pred_{satellite}_{model_type}_{ts}.csv"
            output_file = (UPLOAD_DIR / pred_name)
            df_raw.to_csv(output_file, index=False)
            csv_url = f"{settings.MEDIA_URL}uploads/{pred_name}"

            # JSON-safe results
            results_safe = results_df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")

            # final response
            if row_numbers is None:
                total_rows = pd.read_csv(tmp_path, comment="#").shape[0]
                from_row_resp, to_row_resp = 0, total_rows - 1
            else:
                from_row_resp, to_row_resp = from_resp, to_resp

            return Response({
                "ok": True,
                "satellite": satellite,
                "model_type": model_type,
                "from_row": from_row_resp,
                "to_row": to_row_resp,
                "total_predicted": len(results_df),
                "csv_file": request.build_absolute_uri(csv_url),
                "results": results_safe,
            }, status=status.HTTP_200_OK)

        except FileNotFoundError as e:
            return Response({"ok": False, "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"ok": False, "error": f"Unexpected error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ListUploadsView(APIView):
    def get(self, request):
        files = []
        for p in UPLOAD_DIR.glob("*.csv"):
            files.append(
                {
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "url": request.build_absolute_uri(f"{settings.MEDIA_URL}uploads/{p.name}"),
                }
            )
        return Response({"ok": True, "count": len(files), "files": files})


class MergeCSVView(APIView):
    def post(self, request):
        is_multipart = request.content_type and "multipart/form-data" in request.content_type
        is_json = request.content_type and "application/json" in request.content_type

        try:
            if is_multipart:
                up_a = request.FILES.get("file_a")
                up_b = request.FILES.get("file_b")

                dedupe = str(request.POST.get("dedupe", "true")).lower() in {"1", "true", "yes", "on"}
                output_name = (request.POST.get("output_name") or "").strip() or None

                if up_a and up_b:
                    stored_a = _save_upload_to_media(up_a)   # saved in uploads/
                    stored_b = _save_upload_to_media(up_b)
                    file_a = stored_a
                    file_b = stored_b
                else:
                    file_a = (request.POST.get("file_a") or "").strip()
                    file_b = (request.POST.get("file_b") or "").strip()
                    if not file_a or not file_b:
                        return Response(
                            {"ok": False, "error": "Provide (file_a & file_b) as files OR filenames"},
                            status=400,
                        )

            elif is_json:
                payload = json.loads(request.body.decode("utf-8"))
                file_a = (payload.get("file_a") or "").strip()
                file_b = (payload.get("file_b") or "").strip()
                output_name = (payload.get("output_name") or "").strip() or None
                dedupe = bool(payload.get("dedupe", True))

                if not file_a or not file_b:
                    return Response({"ok": False, "error": "file_a and file_b are required"}, status=400)

            else:
                return Response({"ok": False, "error": "Use multipart/form-data or application/json"}, status=415)

            # 👉 Save merged file into MERGE_DIR instead of uploads
            out_path, total_rows = merge_csvs(
                file_a, file_b, dedupe=dedupe, output_name=output_name, output_dir=MERGE_DIR
            )

            # Build rel media path (mergefiles/<filename>)
            rel_media_path = str(out_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
            merged_url = _abs_media_url(request, rel_media_path)
            token = _make_merge_token(rel_media_path)

            return Response({
                "ok": True,
                "message": "Merge complete",
                "merged_filename": out_path.name,
                "merged_rows": int(total_rows),
                "merged_url": merged_url,      
                "merge_token": token,
                "expires_in": 1800,
            }, status=200)

        except Exception as e:
            return Response({"ok": False, "error": str(e)}, status=500)

        except FileNotFoundError as e:
            return Response({"ok": False, "error": str(e)}, status=404)
        except PermissionError as e:
            return Response({"ok": False, "error": str(e)}, status=403)
        except ValueError as e:
            return Response({"ok": False, "error": str(e)}, status=400)
        except OSError as e:
            return Response({"ok": False, "error": f"I/O error: {e}"}, status=500)
        except Exception as e:
            return Response({"ok": False, "error": f"Unexpected error: {e}"}, status=500)
        

    def get(self, request):
        """
        GET /api/merge
        Returns the list of available merged CSVs from MEDIA_ROOT/mergefiles.
        """
        from pathlib import Path
        import datetime

        # Ensure MERGE_DIR exists (defined globally in your file)
        files = []
        for p in MERGE_DIR.glob("*.csv"):
            try:
                rel_media_path = str(p.relative_to(MEDIA_ROOT)).replace("\\", "/")
            except Exception:
                rel_media_path = f"mergefiles/{p.name}"

            url = _abs_media_url(request, rel_media_path)
            token = _make_merge_token(rel_media_path)


            try:
                with open(p, "rb") as fh:
                    # count lines; subtract 1 for header if file has at least 1 line
                    total_lines = sum(1 for _ in fh)
                rows = max(0, total_lines - 1)
            except Exception:
                rows = None

            stat = p.stat()
            files.append({
                "filename": p.name,
                "url": url,
                "rows": rows,
                "size_bytes": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "merge_token": token,
                "rel_path": rel_media_path,
            })

        # Sort by modified desc
        files.sort(key=lambda x: x["modified"], reverse=True)

        return Response({
            "ok": True,
            "count": len(files),
            "files": files,
        }, status=200)



#  (OPTIONAL) MERGE+TRAIN
class MergeAndTrainView(APIView):
    """
    Backward-compatible helper: merge two CSVs, then best-effort start training.
    Prefer calling /api/merge first, then /api/train with the merged file path/token.
    """

    def post(self, request, *args, **kwargs):
        satellite = "K2"
        model_type = "rf"
        dedupe = True
        output_name = None
        file_a_name = None
        file_b_name = None

        try:
            is_json = request.content_type and "application/json" in request.content_type
            if is_json:
                payload = json.loads(request.body.decode("utf-8"))
                file_a_name = (payload.get("file_a") or payload.get("file_a_name") or "").strip()
                file_b_name = (payload.get("file_b") or payload.get("file_b_name") or "").strip()
                satellite = (payload.get("satellite") or satellite).upper().strip()
                model_type = (payload.get("model") or model_type).strip()

                raw = payload.get("dedupe", dedupe)
                dedupe = raw.strip().lower() in ("1", "true", "yes", "on") if isinstance(raw, str) else bool(raw)
                output_name = (payload.get("output_name") or "").strip() or None
            else:
                satellite = (request.POST.get("satellite") or satellite).upper().strip()
                model_type = (request.POST.get("model") or model_type).strip()
                dv = request.POST.get("dedupe")
                if dv is not None:
                    dedupe = str(dv).lower() in ("1", "true", "yes", "on")
                output_name = (request.POST.get("output_name") or "").strip() or None

                file_a_name = (request.POST.get("file_a_name") or request.POST.get("file_a") or "").strip()
                file_b_name = (request.POST.get("file_b_name") or request.POST.get("file_b") or "").strip()

                # Allow files too
                file_a_file = request.FILES.get("file_a_file")
                file_b_file = request.FILES.get("file_b_file")
                if file_a_file:
                    file_a_name = _save_upload_to_media(file_a_file)
                if file_b_file:
                    file_b_name = _save_upload_to_media(file_b_file)

            if not file_a_name or not file_b_name:
                return Response(
                    {
                        "ok": True,
                        "message": (
                            "Please provide two CSVs to merge "
                            "(JSON: file_a & file_b) or "
                            "(form-data: file_a_name & file_b_name) or "
                            "(files: file_a_file & file_b_file)."
                        ),
                        "merged_file": None,
                        "merged_url": None,
                        "rows": 0,
                        "job_started": False,
                        "job_id": None,
                        "status": "NOT_STARTED",
                        "status_url": None,
                        "logs_url": None,
                    },
                    status=status.HTTP_200_OK,
                )

            # Merge
            out_path, total_rows = merge_csvs(file_a_name, file_b_name, dedupe=dedupe, output_name=output_name)
            merged_url = request.build_absolute_uri(f"{settings.MEDIA_URL}uploads/{out_path.name}")

            # Training (best-effort)
            job_id = None
            job_started = False
            try:
                data_path = str(out_path).replace("\\", "/")
                job_info = create_training_job(data_path=data_path, satellite=satellite, model_type=model_type)
                job_id = job_info.get("job_id")
                job_started = bool(job_id)
            except Exception:
                job_id = None
                job_started = False

            status_url = request.build_absolute_uri(f"/api/train/{job_id}/status") if job_started and job_id else None
            logs_url = request.build_absolute_uri(f"/api/train/{job_id}/logs") if job_started and job_id else None

            return Response(
                {
                    "ok": True,
                    "message": "Merged successfully" + ("; training started" if job_started else "; training could not be started"),
                    "merged_file": out_path.name,
                    "merged_url": merged_url,
                    "rows": total_rows,
                    "job_started": job_started,
                    "job_id": job_id,
                    "status": "PENDING" if job_started else "MERGED_ONLY",
                    "status_url": status_url,
                    "logs_url": logs_url,
                    "params": {
                        "satellite": satellite,
                        "model": model_type,
                        "data_path": str(out_path).replace("\\", "/"),
                        "dedupe": dedupe,
                        "output_name": output_name,
                    },
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {
                    "ok": True,
                    "message": "Merge request processed with errors",
                    "error": str(e),
                    "merged_file": None,
                    "merged_url": None,
                    "rows": 0,
                    "job_started": False,
                    "job_id": None,
                    "status": "ERROR_HANDLED",
                    "status_url": None,
                    "logs_url": None,
                },
                status=status.HTTP_200_OK,
            )
