from django import forms

class UploadFileForm(forms.Form):
    job_name = forms.CharField(max_length=200)
    file = forms.FileField()
