from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils import timezone
import progres as pg

from .models import Submission
from .forms import UploadFileForm

pg_model = pg.load_trained_model()

def read_pdb_coords(f):
    s = f.read().decode("utf-8")
    coords = []
    for line in s.split("\n"):
        if (line.startswith("ATOM  ") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL"):
            break
    return coords

def index(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            coords = read_pdb_coords(request.FILES["file"])
            embedding = pg.embed_coords(coords, model=pg_model)
            submission = Submission(
                job_name=form.cleaned_data["job_name"],
                n_residues=len(coords),
                embedding=embedding.tolist(),
                submission_time=timezone.now(),
            )
            submission.save()
            return HttpResponseRedirect(reverse("progres_search:results", args=(submission.id,)))
    else:
        form = UploadFileForm()
    return render(request, "progres_search/index.html", {"form": form})

def results(request, submission_id):
    submission = get_object_or_404(Submission, pk=submission_id)
    return render(request, "progres_search/results.html", {"submission": submission})
