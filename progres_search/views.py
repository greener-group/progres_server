from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
import progres as pg

from .models import Submission

pg_model = pg.load_trained_model()

def index(request):
    return render(request, "progres_search/index.html")

def results(request, submission_id):
    submission = get_object_or_404(Submission, pk=submission_id)
    return render(request, "progres_search/results.html", {"submission": submission})
