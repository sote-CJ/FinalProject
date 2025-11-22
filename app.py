# app.py (변경 버전에서 핵심 부분만)

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import pandas as pd

from core.thermoeco import run_thermoeco_model

app = Flask(__name__)
app.secret_key = "your-secret-key"   # 세션 사용을 위해 아무 문자열이나

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "capstone2.xlsx")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def input_page():
    # 입력 페이지 (파일 업로드 + 슬라이더만)
    return render_template("input.html")


@app.route("/result")
def result_page():
    # 세션에 저장된 결과 불러오기
    result = session.get("last_result")
    if result is None:
        # 바로 result로 들어오면 다시 입력 페이지로
        return redirect(url_for("input_page"))
    return render_template("result.html", result=result)


# # --- 파일 업로드: 재료 DB (선택, 없으면 capstone2.xlsx 사용) ---
# @app.route("/api/upload_material", methods=["POST"])
# def api_upload_material():
#     file = request.files.get("file")
#     if not file:
#         return jsonify({"ok": False, "error": "파일이 없습니다."}), 400

#     filename = "material_" + file.filename
#     save_path = os.path.join(UPLOAD_DIR, filename)
#     file.save(save_path)

#     return jsonify({"ok": True, "filename": filename})


# --- 파일 업로드: 조건 파일 (Ta,Tb,Tc,Tj) ---
@app.route("/api/upload_condition", methods=["POST"])
def api_upload_condition():
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "error": "파일이 없습니다."}), 400

    filename = "condition_" + file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    return jsonify({"ok": True, "filename": filename})


def load_material_df(material_filename=None):
    # if material_filename:
    #     path = os.path.join(UPLOAD_DIR, material_filename)
    #     if path.lower().endswith(".csv"):
    #         return pd.read_csv(path)
    #     else:
    #         return pd.read_excel(path)
    # else:
    #     return pd.read_excel(DATA_PATH)
    return pd.read_excel(DATA_PATH)


def load_condition_values(condition_filename):
    path = os.path.join(UPLOAD_DIR, condition_filename)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    row = df.iloc[0]
    Ta = float(row["Ta"])
    Tb = float(row["Tb"])
    Tc = float(row["Tc"])
    Tj = float(row["Tj"])
    return Ta, Tb, Tc, Tj



@app.route("/run", methods=["POST"])
def run_analysis():
    data = request.json

    # 업로드된 파일명
    # material_filename = data.get("materialFilename")  # 재료 DB (선택)
    condition_filename = data.get("conditionFilename")  # 조건 파일 (필수)

    if not condition_filename:
        return jsonify({"ok": False, "error": "조건 파일을 업로드해주세요."}), 400

    # ECO 가중치는 슬라이더에서 그대로 받음
    k1 = float(data["k1"])
    k2 = float(data["k2"])
    k3 = float(data["k3"])
    k4 = float(data["k4"])

     # θ 가중치 (사용자 입력)
    ka = float(data.get("ka", 1.0))
    kb = float(data.get("kb", 1.0))
    kt = float(data.get("kt", 1.0))

    # 파일에서 Ta,Tb,Tc,Tj 읽기
    Ta, Tb, Tc, Tj = load_condition_values(condition_filename)

    # 재료 DB 로딩
    # df_material = load_material_df(material_filename)
    df_material = load_material_df()
    # 모델 실행
    result = run_thermoeco_model(
        Ta, Tb, Tc, Tj,
        ka, kb, kt,
        k1, k2, k3, k4,
        df_source=df_material,
    )

    # 세션에 결과 저장 → /result 페이지에서 사용
    session["last_result"] = result

    return jsonify({"ok": True, "redirect": url_for("result_page")})


if __name__ == "__main__":
    app.run(debug=True)
