# plots/urls.py
from django.urls import path
from . import views

app_name = "plots"

urlpatterns = [
    path("", views.plot_page, name="plot_page"),     # 템플릿 페이지
    path("plot.png", views.plot_png, name="plot_png")# PNG 이미지 엔드포인트
]
