from django.urls import include, path

from rest_framework import routers

from api.views import FilesViewSet

router = routers.DefaultRouter()
router.register(r'files', FilesViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
