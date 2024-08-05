from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import progres as pg

class Submission(models.Model):
    targetdbs   = [(x, x) for x in pg.pre_embedded_dbs_faiss + pg.pre_embedded_dbs]
    fileformats = [(x, x) for x in ["guess", "pdb", "mmcif", "mmtf"]]

    job_name = models.CharField(max_length=200, blank=True)
    n_res_total = models.IntegerField()
    res_ranges = models.CharField(max_length=200)
    dom_pdbs = models.JSONField()
    embeddings = models.JSONField()
    targetdb = models.CharField(
        max_length=20,
        choices=targetdbs,
        default="afted",
        verbose_name="Target database",
        help_text="Choose a database to search against",
    )
    fileformat = models.CharField(
        max_length=20,
        choices=fileformats,
        default="guess",
        verbose_name="File format",
    )
    minsimilarity = models.FloatField(
        default=0.8,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        verbose_name="Minimum similarity",
    )
    maxhits = models.IntegerField(
        default=100,
        validators=[MinValueValidator(1), MaxValueValidator(1000)],
        verbose_name="Max number of hits",
    )
    chainsaw = models.BooleanField(default=False, verbose_name="Split domains")
    submission_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.job_name
