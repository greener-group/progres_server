# Progres web server

This repository contains the code for the Progres web server.
The server is hosted at [progres.mrc-lmb.cam.ac.uk](https://progres.mrc-lmb.cam.ac.uk) and the underlying software is available [here](https://github.com/greener-group/progres).
The code here will only be of use if you want to host your own version of the web server.

If you use the Progres web server or software, please cite the paper:

- Greener JG and Jamali K. Fast protein structure searching using structure graph embeddings. bioRxiv (2022) - [link](https://www.biorxiv.org/content/10.1101/2022.11.28.518224)

## Installation

1. Python 3.8 or later is required. The software is OS-independent.
2. Install [Progres](https://github.com/greener-group/progres) v0.2.5 or later as described in the readme.
3. Install [Django](https://www.djangoproject.com) with `pip install Django`.
4. Download this repository with `git clone https://github.com/greener-group/progres_server.git`.
5. Set up the database:
```bash
cd progres_server
python manage.py makemigrations progres_search
python manage.py migrate
```
6. Run the server with `python manage.py runserver`.

To add a dummy submission from `python manage.py shell`:
```
from progres_search.models import Submission
from django.utils import timezone
import torch
from random import random

Submission.objects.all()
s1 = Submission(
    url_str="ABC123",
    job_name="job 1",
    n_res_total=500,
    res_ranges="1-100,201-300_301-400",
    dom_pdbs=["", ""],
    embeddings=[torch.rand(128).tolist(), torch.rand(128).tolist()],
    targetdb="afted",
    chainsaw=True,
    minsimilarity=0.8,
    maxhits=100,
    fileformat="guess",
    submission_time=timezone.now(),
)
s1.save()
Submission.objects.all()
```
To delete the database when changing models:
```
rm db.sqlite3 progres_search/migrations/*.py progres_search/migrations/__pycache__/*
```
