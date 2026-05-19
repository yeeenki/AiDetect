from django.urls import path, include
from django.shortcuts import render, redirect
from .views import register, user_login, user_home, user_logout, detect_ai, detect_api, history_view,detect_audio,detector_audio_page,detect_media_api

urlpatterns = [
    path('', user_home, name='home'),
    path('login/', user_login, name='login'),
    path('register/', register, name="register"),
    path('logout/', user_logout, name = 'logout'),
    path('detect/', detect_ai, name="detect_ai"),
    path('detect/api/', detect_api, name="detect_api"),
    path('history/', history_view, name='history'),
    path('audio/', detector_audio_page, name="detect_audio_page"),
    path('audio/predict/', detect_audio, name="detect_audio_api"),
    path('media/predict/', detect_media_api, name="detect_media_api"),
 
]