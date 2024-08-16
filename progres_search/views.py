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
from random import choices
import string
from tempfile import NamedTemporaryFile

from .models import Submission
from .forms import SubmitJobForm

base_url = "progres.mrc-lmb.cam.ac.uk"
example_url_str = "ABC123"
device = "cpu"

print("Loading Progres data, this will take a minute")

data_dir = os.path.join(os.path.dirname(__file__), "data")

pg_model = pg.load_trained_model(device)
target_data_dict = {}
for targetdb in (pg.pre_embedded_dbs_faiss + pg.pre_embedded_dbs):
    fp = os.path.join(pg.database_dir, targetdb + ".pt")
    target_data_dict[targetdb] = torch.load(fp, map_location=device)
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

def read_ca_backbone(fp, fileformat="guess", res_ranges="all"):
    chosen_format = pg.get_file_format(fp, fileformat)
    if res_ranges == "all":
        n_domains = 1
    else:
        domains_res = []
        for res_range in res_ranges.split(","):
            domain_res = []
            for rr in res_range.split("_"):
                domain_res.extend(pg.extract_res_range(rr))
            domains_res.append(set(domain_res))
            n_domains = len(domains_res)

    dom_coords_ca, dom_pdbs = [[] for _ in range(n_domains)], ["" for _ in range(n_domains)]
    n_res_total = 0

    if chosen_format == "pdb":
        with open(fp) as f:
            chain_id = None
            for line in f.readlines():
                if line.startswith("ATOM  "):
                    if chain_id is None:
                        chain_id = line[21]
                    elif line[21] != chain_id:
                        break # Only read first chain
                    resnum = int(line[22:26])
                    atom_name = line[12:16].strip()
                    if atom_name == "CA":
                        n_res_total += 1
                        for di in range(n_domains):
                            if res_ranges == "all" or resnum in domains_res[di]:
                                dom_coords_ca[di].append(read_pdb_coords(line))
                                break
                    for di in range(n_domains):
                        if res_ranges == "all" or resnum in domains_res[di]:
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
                chain_id = chain.get_id()[0] # Restrict chain ID to be one character
                for res in chain:
                    if res.get_id()[0] == " ": # Ignore hetero atoms
                        resnum, ins_code = res.get_id()[1], res.get_id()[2]
                        resname = res.get_resname()
                        for a in res:
                            if a.get_name() == "CA":
                                n_res_total += 1
                                for di in range(n_domains):
                                    if res_ranges == "all" or resnum in domains_res[di]:
                                        x, y, z = a.get_coord()
                                        dom_coords_ca[di].append([float(x), float(y), float(z)])
                                        break
                            for di in range(n_domains):
                                if res_ranges == "all" or resnum in domains_res[di]:
                                    x, y, z = a.get_coord()
                                    pdb_line = f"ATOM  {a.get_serial_number():>5} {a.get_fullname():4}{a.get_altloc():1}{resname:3} {chain_id:1}{resnum:>4}{ins_code:1}   {x:8.3f}{y:8.3f}{z:8.3f}{a.get_occupancy():6.2f}{a.get_bfactor():6.2f}              \n"
                                    dom_pdbs[di] += pdb_line
                                    break
                break # Only read first chain
            break # Only read first model
    else:
        raise ValueError("fileformat must be \"guess\", \"pdb\", \"mmcif\" or \"mmtf\"")
    return dom_coords_ca, dom_pdbs, n_res_total

def generate_url_str():
    done = False
    while not done:
        trial_url_str = "".join(choices(string.ascii_uppercase + string.digits, k=6))
        if len(Submission.objects.filter(url_str=trial_url_str)) == 0:
            done = True
    return trial_url_str

def index(request):
    if request.method == "POST":
        form = SubmitJobForm(request.POST, request.FILES)
        if form.is_valid():
            fileformat = form.cleaned_data["fileformat"]
            chainsaw = form.cleaned_data["chainsaw"]
            # Keep the file name ending to allow the file format to be guessed
            try:
                temp_file = NamedTemporaryFile(suffix=("." + request.FILES["file"].name))
                with open(temp_file.name, "wb+") as destination:
                    for chunk in request.FILES["file"].chunks():
                        destination.write(chunk)
            except:
                error_text = "Error uploading file. Maybe try again later."
                return render(request, "progres_search/error.html", {"error_text": error_text})
            if chainsaw:
                try:
                    res_ranges = pg.predict_domains(
                        temp_file.name,
                        pg.get_file_format(temp_file.name, fileformat),
                        device,
                    )
                except:
                    error_text = ("Error running Chainsaw. Check your uploaded file and make "
                                  "sure that it contains protein residues and that the correct "
                                  "file format is selected during upload.")
                    return render(request, "progres_search/error.html", {"error_text": error_text})
                if res_ranges is None:
                    error_text = ("Chainsaw did not find any domains in your uploaded protein "
                                  "structure. Try running without splitting into domains.")
                    return render(request, "progres_search/error.html", {"error_text": error_text})
            else:
                res_ranges = "all"
            try:
                dom_coords_ca, dom_pdbs, n_res_total = read_ca_backbone(temp_file.name,
                                                                        fileformat, res_ranges)
            except:
                error_text = ("Error reading structure file. Check your uploaded file and make "
                              "sure the correct file format is selected during upload.")
                return render(request, "progres_search/error.html", {"error_text": error_text})
            temp_file.close()
            embeddings = [pg.embed_coords(c, model=pg_model, device=device).tolist()
                          for c in dom_coords_ca]
            url_str = generate_url_str()
            submission = Submission(
                url_str=url_str,
                job_name=form.cleaned_data["job_name"],
                n_res_total=n_res_total,
                res_ranges=res_ranges,
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
            return HttpResponseRedirect(reverse("progres_search:results", args=(url_str,)))
    else:
        form = SubmitJobForm()
    context = {"form": form, "example_url_str": example_url_str}
    return render(request, "progres_search/index.html", context)

def get_target_url(hid, note, targetdb):
    if targetdb == "afted":
        afdb_id = note.split()[0]
        return f"https://alphafold.ebi.ac.uk/files/{afdb_id}-model_v4.pdb"
    elif targetdb == "scope95" or targetdb == "scope40":
        pdbid = scope_data[hid][1]
        return f"https://files.rcsb.org/download/{pdbid}.pdb"
    elif targetdb == "cath40":
        pdbid = hid[:4].upper()
        return f"https://files.rcsb.org/download/{pdbid}.pdb"
    elif targetdb == "ecod70":
        pdbid = ecod_data[hid][1]
        return f"https://files.rcsb.org/download/{pdbid}.pdb"
    elif targetdb == "af21org":
        entry_id = hid.split("_")[1]
        return f"https://alphafold.ebi.ac.uk/files/AF-{entry_id}-F1-model_v4.pdb"

def get_res_range(hid, note, targetdb):
    if targetdb == "afted":
        return note.split()[1] + ":A"
    elif targetdb == "scope95" or targetdb == "scope40":
        return scope_data[hid][0]
    elif targetdb == "cath40":
        return cath_data[hid]
    elif targetdb == "ecod70":
        return ecod_data[hid][0]
    elif targetdb == "af21org":
        cols = hid.split("_")
        return f"{cols[2]}-{cols[3]}:A"

def get_domain_size(res_range):
    n_res = 0
    for rr in res_range.split("_"):
        n_res += len(pg.extract_res_range(rr))
    return n_res

def results(request, submission_url_str):
    submission = get_object_or_404(Submission, url_str=submission_url_str)
    targetdb = submission.targetdb
    search_type = "faiss" if targetdb in pg.pre_embedded_dbs_faiss else "torch"
    embs_cat = torch.stack([torch.tensor(emb) for emb in submission.embeddings]).to(device)
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
        "base_url"       : base_url,
        "progres_version": importlib.metadata.version("progres"),
        "chainsaw_str"   : "yes" if submission.chainsaw else "no",
        "faiss_str"      : "yes" if search_type == "faiss" else "no",
    }
    return render(request, "progres_search/results.html", context)
