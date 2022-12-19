from rest_framework import viewsets

from api.serializers import FilesSerializer
from api.models import Files


class FilesViewSet(viewsets.ModelViewSet):
    queryset = Files.objects.all()
    serializer_class = FilesSerializer
