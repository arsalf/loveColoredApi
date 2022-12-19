from rest_framework import serializers

from api.models import Files

class FilesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Files
        fields = ('name', 'file')
