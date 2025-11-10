# plots/views.py
import io
import os
import matplotlib
matplotlib.use("Agg")

from django.conf import settings
from django.http import HttpResponse, Http404
from django.shortcuts import render

from .plot_builder import build_figure

def plot_page(request):
    # 간단한 페이지에서 <img src="{% url 'plots:plot_png' %}">로 그림을 표시
    return render(request, "plots/plot_page.html")

def plot_png(request):
    try:
        # 1) 쿼리 파라미터(미지정 시 기본값)
        Ta = float(request.GET.get("Ta", "25"))
        Tb = float(request.GET.get("Tb", "25"))
        Tc = float(request.GET.get("Tc", "25"))
        Tj = float(request.GET.get("Tj", "100"))

        ka = float(request.GET.get("ka", "33"))
        kb = float(request.GET.get("kb", "33"))
        kt = float(request.GET.get("kt", "34"))

        k1 = float(request.GET.get("k1", "25"))
        k2 = float(request.GET.get("k2", "25"))
        k3 = float(request.GET.get("k3", "25"))
        k4 = float(request.GET.get("k4", "25"))

        # 2) 엑셀 경로 (기본: media/capstone_result.xlsx)
        rel_excel = request.GET.get("excel", "capstone_result.xlsx")
        excel_path = os.path.join(settings.MEDIA_ROOT, rel_excel)
        if not os.path.exists(excel_path):
            raise Http404(f"Excel not found: {rel_excel}")

        # 3) Figure 생성
        fig = build_figure(Ta, Tb, Tc, Tj, ka, kb, kt, k1, k2, k3, k4, excel_path)

        # 4) PNG 직렬화
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        return HttpResponse(buf.getvalue(), content_type="image/png")
    except Http404:
        raise
    except Exception as e:
        return HttpResponse(f"Plot error: {e}", status=500, content_type="text/plain")
