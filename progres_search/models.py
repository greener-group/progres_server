from django.db import models

class Submission(models.Model):
    job_name = models.CharField(max_length=200)
    submission_time = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.job_name
