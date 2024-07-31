from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.mmtf import MMTFParser
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

print("Loading Progres data will take a minute")

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

backbone_atoms = ["N", "CA", "C", "O"]

def read_pdb_coords(line):
    return [float(line[30:38]), float(line[38:46]), float(line[46:54])]

def read_ca_backbone(fp, fileformat="guess"):
    if fileformat == "guess":
        chosen_format = "pdb"
        file_ext = os.path.splitext(fp)[1].lower()
        if file_ext == ".cif" or file_ext == ".mmcif":
            chosen_format = "mmcif"
        elif file_ext == ".mmtf":
            chosen_format = "mmtf"
    else:
        chosen_format = fileformat

    coords_ca, coords_backbone = [], []
    if chosen_format == "pdb":
        with open(fp) as f:
            for line in f.readlines():
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    if atom_name == "CA":
                        coords_ca.append(read_pdb_coords(line))
                    if atom_name in backbone_atoms:
                        coords_backbone.append(read_pdb_coords(line))
                elif line.startswith("ENDMDL"):
                    break
    elif chosen_format == "mmcif" or chosen_format == "mmtf":
        if chosen_format == "mmcif":
            parser = MMCIFParser()
            struc = parser.get_structure("", fp)
        else:
            struc = MMTFParser.get_structure(fp)
        for model in struc:
            for atom in model.get_atoms():
                if atom.get_name() == "CA":
                    cs = atom.get_coord()
                    coords_ca.append([float(cs[0]), float(cs[1]), float(cs[2])])
                if atom.get_name() in backbone_atoms:
                    cs = atom.get_coord()
                    coords_backbone.append([float(cs[0]), float(cs[1]), float(cs[2])])
            break
    else:
        raise ValueError("fileformat must be \"guess\", \"pdb\", \"mmcif\" or \"mmtf\"")
    return coords_ca, coords_backbone

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
            coords_ca, coords_backbone = read_ca_backbone(temp_file.name, fileformat)
            temp_file.close()
            embedding = pg.embed_coords(coords_ca, model=pg_model)
            submission = Submission(
                job_name=form.cleaned_data["job_name"],
                n_residues=len(coords_ca),
                coords_backbone=coords_backbone,
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

def get_target_url(note, targetdb):
    if targetdb == "afted":
        afdb_id = note.split()[0]
        return f"https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v4.pdb"
    return ""

def get_res_range(note, targetdb):
    if targetdb == "afted":
        return note.split()[1]
    return ""

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

    target_urls = [get_target_url(note, targetdb) for note in result_dict["notes"]]
    res_ranges = [get_res_range(note, targetdb) for note in result_dict["notes"]]
    results_zip = zip(result_dict["domains"], result_dict["hits_nres"],
                      result_dict["similarities"], result_dict["notes"],
                      target_urls, res_ranges)
    query_pdb = ""
    for i, (x, y, z) in enumerate(submission.coords_backbone):
        atom_name = backbone_atoms[i % 4]
        res_n = (i // 4) + 1
        query_pdb += f"ATOM  {(i + 1):5}  {atom_name:2}  ALA A{res_n:4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}  \n"
    query_pdb += "END   \n"

    context = {
        "submission"     : submission,
        "results"        : results_zip,
        "progres_version": importlib.metadata.version("progres"),
        "faiss_str"      : ", FAISS search" if search_type == "faiss" else "",
        "query_pdb"      : query_pdb,
    }
    return render(request, "progres_search/results.html", context)
