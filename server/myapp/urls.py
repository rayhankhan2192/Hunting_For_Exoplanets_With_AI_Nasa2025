# server/myapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("api/train", views.start_training, name="start_training"),
    path("api/train/<str:job_id>/status", views.training_status, name="training_status"),
    path("api/train/<str:job_id>/logs", views.training_logs, name="training_logs"),
]
