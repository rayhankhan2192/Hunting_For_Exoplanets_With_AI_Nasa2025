# # from django.urls import path
# # from . import views

# # urlpatterns = [
# #     # Training endpoints
# #     path("api/train", views.start_training, name="start_training"),
# #     path("api/train/<str:job_id>/status", views.training_status, name="training_status"),
# #     path("api/train/<str:job_id>/logs", views.training_logs, name="training_logs"),

# #     # Prediction endpoint
# #     path("api/predict", views.prediction_view, name="prediction_view"),
# #     path("api/uploads", views.list_uploads, name="list_uploads"),  
# #     path("api/merge-train", views.merge_and_train, name="merge_and_train"),
# # ]

# from django.urls import path
# from .views import (
#     StartTrainingView, TrainingStatusView, TrainingLogsView,
#     PredictionView, ListUploadsView, MergeAndTrainView
# )

# urlpatterns = [
#     path("api/train", StartTrainingView.as_view(), name="start_training"),
#     path("api/train/<str:job_id>/status", TrainingStatusView.as_view(), name="training_status"),
#     path("api/train/<str:job_id>/logs", TrainingLogsView.as_view(), name="training_logs"),

#     path("api/predict", PredictionView.as_view(), name="prediction_view"),
#     path("api/uploads", ListUploadsView.as_view(), name="list_uploads"),
#     path("api/merge-train", MergeAndTrainView.as_view(), name="merge_and_train"),
# ]


from django.urls import path
from .views import (
    StartTrainingView, TrainingStatusView, TrainingLogsView,
    PredictionView, ListUploadsView, MergeAndTrainView, MergeCSVView
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
]
