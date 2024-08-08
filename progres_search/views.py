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

device = "cpu"

print("Loading Progres data, this will take a minute")

data_dir = os.path.join(os.path.dirname(__file__), "data")

pg_model = pg.load_trained_model()
target_data_dict = {}
for targetdb in (pg.pre_embedded_dbs_faiss + pg.pre_embedded_dbs):
    fp = os.path.join(pg.database_dir, targetdb + ".pt")
    target_data_dict[targetdb] = torch.load(fp)
target_index_dict = {}
for targetdb in pg.pre_embedded_dbs_faiss:
    fp = os.path.join(pg.database_dir, f"{targetdb}.index")
    target_index_dict[targetdb] = faiss.read_index(fp)

scope_data = {}
with open(os.path.join(data_dir, "dir.des.scope.2.08-stable.txt")) as f:
    for line in f.readlines():
        if not line.startswith("#"):
            cols = line.split()
            if cols[1] == "px":
                dom_id, pdb_id, res_range_scope = cols[3], cols[4], cols[5]
                if res_range_scope != "-":
                    chain_id = res_range_scope.split(":")[0]
                    if res_range_scope.endswith(":"):
                        res_range = "*:" + chain_id
                    else:
                        res_ranges = [rr.split(":")[1] for rr in res_range_scope.split(",")]
                        res_range = "_".join(res_ranges) + ":" + chain_id
                    scope_data[dom_id] = [res_range, pdb_id]

cath_data = {}
with open(os.path.join(data_dir, "cath-b-newest-all")) as f:
    for line in f.readlines():
        cols = line.split()
        dom_id, res_range_cath = cols[0], cols[3]
        res_ranges = [rr.split(":")[0] for rr in res_range_cath.split(",")]
        chain_id = res_range_cath.split(":")[-1]
        res_range = "_".join(res_ranges) + ":" + chain_id
        cath_data[dom_id] = res_range

ecod_data = {}
with open(os.path.join(data_dir, "ecod.latest.F70.domains.txt")) as f:
    for line in f.readlines():
        if not line.startswith("#"):
            cols = line.split()
            dom_id, pdb_id, res_range_ecod = cols[1], cols[4], cols[6]
            res_ranges = [rr.split(":")[1] for rr in res_range_ecod.split(",")]
            chain_id = res_range_ecod.split(":")[0]
            res_range = "_".join(res_ranges) + ":" + chain_id
            ecod_data[dom_id] = [res_range, pdb_id]

print("Loaded Progres data")

backbone_atoms = ["N", "CA", "C", "O"]

def read_pdb_coords(line):
    return [float(line[30:38]), float(line[38:46]), float(line[46:54])]

def read_ca_backbone(fp, fileformat="guess", res_ranges=None):
    if fileformat == "guess":
        chosen_format = "pdb"
        file_ext = os.path.splitext(fp)[1].lower()
        if file_ext == ".cif" or file_ext == ".mmcif":
            chosen_format = "mmcif"
        elif file_ext == ".mmtf":
            chosen_format = "mmtf"
    else:
        chosen_format = fileformat

    if res_ranges is None:
        domains_res = [set(range(1, 10000+1))]
    else:
        domains_res = []
        for res_range in res_ranges.split(","):
            domain_res = []
            for rr in res_range.split("_"):
                res_start, res_end = rr.split("-")
                domain_res.extend(range(int(res_start), int(res_end) + 1))
            domains_res.append(set(domain_res))
    
    n_domains = len(domains_res)
    dom_coords_ca, dom_pdbs = [[] for _ in range(n_domains)], ["" for _ in range(n_domains)]
    n_res_total = 0

    if chosen_format == "pdb":
        with open(fp) as f:
            for line in f.readlines():
                if line.startswith("ATOM  "):
                    atom_name = line[12:16].strip()
                    if atom_name == "CA":
                        n_res_total += 1
                        for di in range(n_domains):
                            if n_res_total in domains_res[di]:
                                dom_coords_ca[di].append(read_pdb_coords(line))
                                break
                    for di in range(n_domains):
                        if n_res_total in domains_res[di]:
                            dom_pdbs[di] += line
                            break
                elif line.startswith("ENDMDL"):
                    break
    elif chosen_format == "mmcif" or chosen_format == "mmtf":
        if chosen_format == "mmcif":
            parser = MMCIFParser()
            struc = parser.get_structure("", fp)
        else:
            struc = MMTFParser.get_structure(fp)
        for model in struc:
            for chain in model:
                for res in chain:
                    if res.id[0] == " ": # Ignore hetero atoms
                        for a in res:
                            if a.get_name() == "CA":
                                n_res_total += 1
                                for di in range(n_domains):
                                    if n_res_total in domains_res[di]:
                                        x, y, z = a.get_coord()
                                        dom_coords_ca[di].append([float(x), float(y), float(z)])
                                        break
                            for di in range(n_domains):
                                if n_res_total in domains_res[di]:
                                    x, y, z = a.get_coord()
                                    pdb_line = f"ATOM  {a.get_serial_number():>5} {a.get_fullname():4}{a.get_altloc():1}{res.get_resname():3} {chain.get_id():1}{res.get_id()[1]:>4}{res.get_id()[2]:1}   {x:8.3f}{y:8.3f}{z:8.3f}{a.get_occupancy():6.2f}{a.get_bfactor():6.2f}              \n"
                                    dom_pdbs[di] += pdb_line
                                    break
            break # Only read first model
    else:
        raise ValueError("fileformat must be \"guess\", \"pdb\", \"mmcif\" or \"mmtf\"")
    return dom_coords_ca, dom_pdbs, n_res_total

def index(request):
    if request.method == "POST":
        form = SubmitJobForm(request.POST, request.FILES)
        if form.is_valid():
            fileformat = form.cleaned_data["fileformat"]
            chainsaw = form.cleaned_data["chainsaw"]
            # Keep the file name ending to allow the file format to be guessed
            temp_file = NamedTemporaryFile(suffix=("." + request.FILES["file"].name))
            with open(temp_file.name, "wb+") as destination:
                for chunk in request.FILES["file"].chunks():
                    destination.write(chunk)
            if chainsaw:
                res_ranges = pg.predict_domains(temp_file.name)
            else:
                res_ranges = None
            dom_coords_ca, dom_pdbs, n_res_total = read_ca_backbone(temp_file.name,
                                                                    fileformat, res_ranges)
            temp_file.close()
            embeddings = [pg.embed_coords(c, model=pg_model).tolist() for c in dom_coords_ca]
            submission = Submission(
                job_name=form.cleaned_data["job_name"],
                n_res_total=n_res_total,
                res_ranges=("all" if res_ranges is None else res_ranges),
                dom_pdbs=dom_pdbs,
                embeddings=embeddings,
                targetdb=form.cleaned_data["targetdb"],
                fileformat=fileformat,
                minsimilarity=form.cleaned_data["minsimilarity"],
                maxhits=form.cleaned_data["maxhits"],
                chainsaw=chainsaw,
                submission_time=timezone.now(),
            )
            submission.save()
            return HttpResponseRedirect(reverse("progres_search:results", args=(submission.id,)))
    else:
        form = SubmitJobForm()
    return render(request, "progres_search/index.html", {"form": form})

def get_target_url(hid, note, targetdb):
    if targetdb == "afted":
        afdb_id = note.split()[0]
        return f"https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v4.pdb"
    elif targetdb == "af21org":
        entry_id = hid.split("_")[1]
        return f"https://alphafold.ebi.ac.uk/files/AF-{entry_id}-F1-model_v4.pdb"
    return ""

def get_res_range(hid, note, targetdb):
    if targetdb == "afted":
        return note.split()[1]
    elif targetdb == "af21org":
        cols = hid.split("_")
        return f"{cols[2]}-{cols[3]}"
    return ""

def get_domain_size(res_range):
    n_res = 0
    for rr in res_range.split("_"):
        res_start, res_end = rr.split("-")
        n_res += int(res_end) - int(res_start) + 1
    return n_res

def results(request, submission_id):
    submission = get_object_or_404(Submission, pk=submission_id)
    targetdb = submission.targetdb
    search_type = "faiss" if targetdb in pg.pre_embedded_dbs_faiss else "torch"
    embs_cat = torch.stack([torch.tensor(emb) for emb in submission.embeddings])
    data_loader = DataLoader(
        pg.EmbeddingDataset(embs_cat),
        batch_size=pg.get_batch_size(search_type == "faiss"),
        shuffle=False,
        num_workers=0,
    )
    result_dicts = list(pg.search_generator_inner(
        data_loader,
        ["?"] * embs_cat.size(0),
        targetdb,
        target_data_dict[targetdb],
        target_index_dict[targetdb] if search_type == "faiss" else None,
        search_type,
        submission.minsimilarity,
        submission.maxhits,
    ))

    results_zips = []
    for result_dict in result_dicts:
        target_urls = [get_target_url(hid, note, targetdb) for hid, note in zip(
                                                result_dict["domains"], result_dict["notes"])]
        target_res_ranges = [get_res_range(hid, note, targetdb) for hid, note in zip(
                                                result_dict["domains"], result_dict["notes"])]
        results_zip = zip(result_dict["domains"], result_dict["hits_nres"],
                          result_dict["similarities"], result_dict["notes"],
                          target_urls, target_res_ranges)
        results_zips.append(results_zip)

    if submission.res_ranges == "all":
        query_res_ranges = [f"1-{submission.n_res_total}"]
    else:
        query_res_ranges = submission.res_ranges.split(",")
    domains_iter = range(1, len(results_zips) + 1)
    domains_zip = zip(
        domains_iter,
        [get_domain_size(rr) for rr in query_res_ranges],
        query_res_ranges,
        results_zips,
    )
    rd_1_id, rd_1_note = result_dicts[0]["domains"][0], result_dicts[0]["notes"][0]
    url_start = get_target_url(rd_1_id, rd_1_note, targetdb)
    res_range_start = get_res_range(rd_1_id, rd_1_note, targetdb)

    context = {
        "submission"     : submission,
        "query_dom_pdbs" : submission.dom_pdbs,
        "n_domains"      : len(domains_iter),
        "domains_iter"   : domains_iter,
        "domains_zip"    : domains_zip,
        "url_start"      : url_start,
        "res_range_start": res_range_start,
        "progres_version": importlib.metadata.version("progres"),
        "chainsaw_str"   : "yes" if submission.chainsaw else "no",
        "faiss_str"      : "yes" if search_type == "faiss" else "no",
    }
    return render(request, "progres_search/results.html", context)
