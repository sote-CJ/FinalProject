import io, os
import matplotlib
matplotlib.use("Agg")

from django.conf import settings
from django.http import HttpResponse, Http404
from django.shortcuts import render

from .plot_builder import build_figure

def plot_page(request):
    return render(request, "plots/plot_page.html")

def _safe_excel_path(name: str) -> str:
    # 파일명만 허용(디렉터리 구분자/상위경로 차단)
    base = (name or "capstone_result.xlsx").strip()
    base = os.path.normpath(base).replace("\\", "/")
    if "/" in base or base.startswith("../"):
        raise Http404("Invalid excel file name")
    return os.path.join(settings.MEDIA_ROOT, base)

def plot_png(request):
    try:
        # 1) 파라미터
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

        # 2) 엑셀 경로(안전/검증)
        rel_excel = request.GET.get("excel", "capstone_result.xlsx")
        excel_path = _safe_excel_path(rel_excel)
        if not os.path.exists(excel_path):
            raise Http404(f"Excel not found: {rel_excel}")

        # 3) Figure 생성
        fig = build_figure(Ta, Tb, Tc, Tj, ka, kb, kt, k1, k2, k3, k4, excel_path)

        # 4) PNG 응답(고해상도 + 캐시무효화)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor="white")
        buf.seek(0)
        resp = HttpResponse(buf.getvalue(), content_type="image/png")
        resp["Cache-Control"] = "no-store, must-revalidate"
        return resp

    except Http404:
        raise
    except Exception as e:
        return HttpResponse(f"Plot error: {type(e).__name__}: {e}", status=500, content_type="text/plain")
