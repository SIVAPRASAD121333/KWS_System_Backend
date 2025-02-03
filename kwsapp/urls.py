# kwsapp/urls.py
from django.urls import path
from .views import display_page,display_page_bng,display_page_man,display_page_miz, upload_audio,process_audio_bng,process_audio_man,process_audio_miz

urlpatterns = [
    path('', display_page, name='display_page'),
    path('display_page_bng', display_page_bng, name='display_page_bng'),
    path('display_page_man', display_page_man, name='display_page_man'),
    path('display_page_miz', display_page_miz, name='display_page_miz'),
    path('process_audio_miz/', process_audio_miz, name='process_audio_miz'),
    path('process_audio_man/', process_audio_man, name='process_audio_man'),
    path('process_audio_bng/', process_audio_bng, name='process_audio_bng'),
]
