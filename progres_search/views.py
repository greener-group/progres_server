from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils import timezone
import faiss
import progres as pg
import torch
from torch_geometric.loader import DataLoader
import importlib.metadata
import os
from tempfile import NamedTemporaryFile

from .models import Submission
from .forms import SubmitJobForm

pg_model = pg.load_trained_model()
target_data_dict = {}
for targetdb in (pg.pre_embedded_dbs_faiss + pg.pre_embedded_dbs):
    fp = os.path.join(pg.database_dir, targetdb + ".pt")
    target_data_dict[targetdb] = torch.load(fp)
target_index_dict = {}
for targetdb in pg.pre_embedded_dbs_faiss:
    fp = os.path.join(pg.database_dir, f"{targetdb}.index")
    target_index_dict[targetdb] = faiss.read_index(fp)

print("Loaded Progres data")

def index(request):
    if request.method == "POST":
        form = SubmitJobForm(request.POST, request.FILES)
        if form.is_valid():
            fileformat = form.cleaned_data["fileformat"]
            # Keep the file name ending to allow the file format to be guessed
            temp_file = NamedTemporaryFile(suffix=("." + request.FILES["file"].name))
            with open(temp_file.name, "wb+") as destination:
                for chunk in request.FILES["file"].chunks():
                    destination.write(chunk)
            coords = pg.read_coords(temp_file.name, fileformat)
            temp_file.close()
            embedding = pg.embed_coords(coords, model=pg_model)
            submission = Submission(
                job_name=form.cleaned_data["job_name"],
                n_residues=len(coords),
                embedding=embedding.tolist(),
                targetdb=form.cleaned_data["targetdb"],
                fileformat=fileformat,
                minsimilarity=form.cleaned_data["minsimilarity"],
                maxhits=form.cleaned_data["maxhits"],
                submission_time=timezone.now(),
            )
            submission.save()
            return HttpResponseRedirect(reverse("progres_search:results", args=(submission.id,)))
    else:
        form = SubmitJobForm()
    return render(request, "progres_search/index.html", {"form": form})

def results(request, submission_id):
    submission = get_object_or_404(Submission, pk=submission_id)
    targetdb = submission.targetdb
    data_loader = DataLoader(
        pg.EmbeddingDataset(torch.tensor(submission.embedding)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    search_type = "faiss" if targetdb in pg.pre_embedded_dbs_faiss else "torch"
    result_dict = list(pg.search_generator_inner(
        data_loader,
        ["?"],
        targetdb,
        target_data_dict[targetdb],
        target_index_dict[targetdb] if search_type == "faiss" else None,
        search_type,
        submission.minsimilarity,
        submission.maxhits,
    ))[0]
    results_zip = zip(result_dict["domains"], result_dict["hits_nres"],
                      result_dict["similarities"], result_dict["notes"])
    context = {
        "submission"     : submission,
        "results"        : results_zip,
        "progres_version": importlib.metadata.version("progres"),
        "faiss_str"      : ", FAISS search" if search_type == "faiss" else "",
    }
    return render(request, "progres_search/results.html", context)
