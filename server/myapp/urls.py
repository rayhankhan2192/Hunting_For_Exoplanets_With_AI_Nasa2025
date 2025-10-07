from django.urls import path
from .views import (
    StartTrainingView, TrainingStatusView, TrainingLogsView,
    PredictionView, ListUploadsView, MergeAndTrainView, MergeCSVView,
    TrainAllView
)

urlpatterns = [
    # Training endpoints
    path("api/train", StartTrainingView.as_view(), name="start_training"),
    path("api/train/<str:job_id>/status", TrainingStatusView.as_view(), name="training_status"),
    path("api/train/<str:job_id>/logs", TrainingLogsView.as_view(), name="training_logs"),

    # Prediction endpoint
    path("api/predict", PredictionView.as_view(), name="prediction_view"),

    # Uploads & Merge+Train
    path("api/get-uploads-file", ListUploadsView.as_view(), name="list_uploads"),
    path("api/merge-train", MergeAndTrainView.as_view(), name="merge_and_train"),

    path("api/merge", MergeCSVView.as_view(), name="merge_csv"),

    path("api/trainall", TrainAllView.as_view(), name="trainall"),
]
