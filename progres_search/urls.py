from django.urls import path

from . import views

app_name = "progres_search"
urlpatterns = [
    path("", views.index, name="index"),
    path("results/<str:submission_url_str>", views.results, name="results"),
]
