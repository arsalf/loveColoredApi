from rest_framework import serializers

from api.models import Person, Species, Files


class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = ('name', 'birth_year', 'eye_color', 'species')


class SpeciesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Species
        fields = ('name', 'classification', 'language')


class FilesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Files
        fields = ('name', 'file')
