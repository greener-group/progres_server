from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import progres as pg

class Submission(models.Model):
    targetdbs   = [(x, x) for x in pg.pre_embedded_dbs_faiss + pg.pre_embedded_dbs]
    fileformats = [(x, x) for x in ["guess", "pdb", "mmcif", "mmtf"]]

    job_name = models.CharField(
        max_length=200,
        blank=True,
        help_text="Give the job a name (optional).",
    )
    n_res_total = models.IntegerField()
    res_ranges = models.CharField(max_length=200)
    dom_pdbs = models.JSONField()
    embeddings = models.JSONField()
    targetdb = models.CharField(
        max_length=20,
        choices=targetdbs,
        default="afted",
        verbose_name="Target database",
        help_text=("Choose a database to search against. afted is the "
                   "<a href='https://www.biorxiv.org/content/10.1101/2024.03.18.585509'>"
                   "TED domains</a> from the AlphaFold database. scope95/scope40/cath40/ecod70 "
                   "are domains from classifications of the PDB. af21org is domains from the "
                   "AlphaFold set of 21 model organisms."),
    )
    chainsaw = models.BooleanField(
        default=False,
        verbose_name="Split domains",
        help_text=("Whether to split the query structure into domains with "
                   "<a href='https://doi.org/10.1093/bioinformatics/btae296'>Chainsaw</a> and "
                   "search with each domain separately. Recommended for structures above "
                   "200-300 residues. Note that whether or not this option is selected, only "
                   "the first chain in the file is considered."),
    )
    minsimilarity = models.FloatField(
        default=0.8,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="Minimum similarity",
        help_text=("The Progres score above which to return hits. The default of 0.8 indicates "
                   "the same fold. Must be 0 -> 1."),
    )
    maxhits = models.IntegerField(
        default=100,
        validators=[MinValueValidator(1), MaxValueValidator(1000)],
        verbose_name="Max number of hits",
        help_text="The maximum number of hits per domain to return. Must be 1 -> 1000.",
    )
    fileformat = models.CharField(
        max_length=20,
        choices=fileformats,
        default="guess",
        verbose_name="File format",
        help_text=("By default the format of the uploaded file is guessed from the file "
                   "extension, but it can be set explicitly here. Supported formats are "
                   "PDB, mmCIF and MMTF."),
    )
    submission_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.job_name
