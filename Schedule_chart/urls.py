"""production_plan_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from .views import sc_view, sc_data, static_json, fjsp_view, model_view
urlpatterns = [
    path('',sc_view),
    path('sc_data/', sc_data),
    path('static_json/', static_json),
    path('schedule_machine/', fjsp_view),
    path('model_cadila/', model_view),
]

