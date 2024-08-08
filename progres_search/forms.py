from django import forms
from django.core.validators import ValidationError

from .models import Submission

def file_size_validator(value):
    limit = 10 * 1024 * 1024
    if value.size > limit:
        raise ValidationError("File too large. Size should not exceed 10 MiB.")

class SubmitJobForm(forms.ModelForm):
    file = forms.FileField(validators=[file_size_validator])

    class Meta:
        model = Submission
        fields = ["job_name", "targetdb", "chainsaw", "minsimilarity", "maxhits", "fileformat"]
