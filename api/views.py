from rest_framework import viewsets

from api.serializers import PersonSerializer, SpeciesSerializer, FilesSerializer
from api.models import Person, Species, Files

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser


class PersonViewSet(viewsets.ModelViewSet):
    queryset = Person.objects.all()
    serializer_class = PersonSerializer


class SpeciesViewSet(viewsets.ModelViewSet):
    queryset = Species.objects.all()
    serializer_class = SpeciesSerializer


class FilesViewSet(viewsets.ModelViewSet):
    queryset = Files.objects.all()
    print(queryset)
    serializer_class = FilesSerializer
