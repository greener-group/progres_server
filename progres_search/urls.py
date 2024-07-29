from django.urls import path

from . import views

app_name = "progres_search"
urlpatterns = [
    path("", views.index, name="index"),
    path("results/<int:submission_id>", views.results, name="results"),
]
