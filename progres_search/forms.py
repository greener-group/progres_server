from django import forms

from .models import Submission

class SubmitJobForm(forms.ModelForm):
    file = forms.FileField()

    class Meta:
        model = Submission
        fields = ["job_name", "targetdb", "chainsaw", "minsimilarity", "maxhits", "fileformat"]
