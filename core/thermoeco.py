# core/thermoeco.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

P = 1.0  # 소비전력 1W 고정


def min_max_norm(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    x_min = x.min()
    x_max = x.max()
    denom = max(x_max - x_min, eps)
    return (x - x_min) / denom


def min_max_norm_invert(x, eps=1e-9):
    s = min_max_norm(x, eps)
    return 1.0 - s


def spread_for_plot(x):
    """0~1 값이지만, 그래프용 막대길이를 0~1 전체를 다 쓰지 않게 펴주는 함수 (0.1~0.9 정도)."""
    x = np.asarray(x, dtype=float)
    x_min = np.min(x)
    x_max = np.max(x)
    return 0.8 * (x - x_min + 0.1) / (x_max - x_min + 1e-9)


def run_thermoeco_model(
    Ta, Tb, Tc, Tj,
    ka, kb, kt,     # Theta 가중치
    k1, k2, k3, k4, # ESG 가중치 (CO2, life, recycle, cost)
    df_source: pd.DataFrame
):
    """middlepj(1)의 전체 로직을 실행하고,
    웹에서 바로 쓸 수 있는 dict 형태로 결과를 반환.
    """

    # ---------------------------
    # 1. 가중치 정규화
    # ---------------------------
    sum_kt = ka + kb + kt
    wa, wb, wt = ka / sum_kt, kb / sum_kt, kt / sum_kt

    sum_ke = k1 + k2 + k3 + k4
    w1, w2, w3, w4 = k1 / sum_ke, k2 / sum_ke, k3 / sum_ke, k4 / sum_ke

    # ---------------------------
    # 2. theta 계산
    # ---------------------------
    theta_ja_in = (Tj - Ta) / P
    theta_jb_in = (Tj - Tb) / P
    theta_jt_in = (Tj - Tc) / P

    # ---------------------------
    # 3. KNN용 데이터 전처리
    # ---------------------------
    df_knn = df_source.dropna(subset=["theta_ja", "theta_jb", "theta_jt"]).copy()
    if len(df_knn) == 0:
        raise ValueError("theta_ja, theta_jb, theta_jt 값이 있는 행이 없습니다.")

    X = df_knn[["theta_ja", "theta_jb", "theta_jt"]].to_numpy()

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    range_ = np.maximum(X_max - X_min, 1e-9)

    X_norm = (X - X_min) / range_
    Y = np.array([theta_ja_in, theta_jb_in, theta_jt_in])
    Y_norm = (Y - X_min) / range_

    df_knn["theta_ja_norm"] = X_norm[:, 0]
    df_knn["theta_jb_norm"] = X_norm[:, 1]
    df_knn["theta_jt_norm"] = X_norm[:, 2]

    # 가중치
    w_theta = np.array([wa, wb, wt])
    sqrt_w = np.sqrt(w_theta)

    X_w = X_norm * sqrt_w
    Y_w = Y_norm * sqrt_w

    # ---------------------------
    # 4. KNN
    # ---------------------------
    n_neighbors = min(5, len(X_w))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(X_w)
    distances, indices = nbrs.kneighbors(Y_w.reshape(1, -1))

    top_idx = indices[0]
    df_top5 = df_knn.iloc[top_idx].copy()
    df_top5["distance"] = distances.flatten()

    # ---------------------------
    # 5. ECO SCORE
    # ---------------------------
    # 5-1) CO2
    theta_ja_mat = df_top5["theta_ja"].to_numpy()
    denom1 = np.maximum(theta_ja_mat, 1e-9)
    co2_ratio = theta_ja_in / denom1
    df_top5["score_co2"] = min_max_norm(co2_ratio)

    # 5-2) 수명
    tg = df_top5["Tg"].to_numpy()
    cte = df_top5["cte"].to_numpy()
    tg_norm = min_max_norm(tg)
    cte_norm = min_max_norm_invert(cte)
    score_life = 0.7 * tg_norm + 0.3 * cte_norm
    df_top5["score_life"] = score_life

    # 5-3) 재활용
    recycle_map = {"A": 1.0, "B": 0.67, "C": 0.33, "D": 0.0}
    df_top5["score_recycle"] = df_top5["recycle"].map(recycle_map).fillna(0.0)

    # 5-4) 가격
    cost = df_top5["cost"].to_numpy()
    df_top5["score_cost"] = min_max_norm_invert(cost)

    # 5-5) 최종 eco
    df_top5["score_eco"] = (
        w1 * df_top5["score_co2"]
        + w2 * df_top5["score_life"]
        + w3 * df_top5["score_recycle"]
        + w4 * df_top5["score_cost"]
    )

    # ---------------------------
    # 6. Thermal + eco 통합점수
    # ---------------------------
    distance_thermal = df_top5["distance"].to_numpy()
    df_top5["score_thermal"] = min_max_norm_invert(distance_thermal)
    df_top5["score_final"] = 0.5 * df_top5["score_thermal"] + 0.5 * df_top5["score_eco"]

    df_top5_ranked = df_top5.sort_values(by="score_final", ascending=False).reset_index(drop=True)

    # ---------------------------
    # 7. 웹에 내려줄 데이터 정리
    # ---------------------------
    # (1) 3D 전체 포인트
    all_points = {
        "x": X_w[:, 0].tolist(),
        "y": X_w[:, 1].tolist(),
        "z": X_w[:, 2].tolist(),
    }

    # (2) 쿼리 포인트
    query_point = {
        "x": float(Y_w[0]),
        "y": float(Y_w[1]),
        "z": float(Y_w[2]),
    }

    # (3) neighbor 포인트
    neighbor_points = {
        "x": X_w[top_idx, 0].tolist(),
        "y": X_w[top_idx, 1].tolist(),
        "z": X_w[top_idx, 2].tolist(),
        "names": df_top5["name"].tolist(),
    }

    # (4) query–neighbor 연결선(Plotly에서 line trace 생성용)
    lines = []
    for i in range(len(top_idx)):
        lines.append({
            "x": [query_point["x"], neighbor_points["x"][i]],
            "y": [query_point["y"], neighbor_points["y"][i]],
            "z": [query_point["z"], neighbor_points["z"][i]],
        })

    # (5) TOP 3 테이블
    top3 = df_top5_ranked.head(3).copy()
    top3["LANK"] = np.arange(1, len(top3) + 1)
    top3["SCORE"] = (top3["score_final"] * 100).round().astype(int)

    top3_table = [
        {
            "lank": int(row["LANK"]),
            "name": row["name"],
            "score": int(row["SCORE"]),
        }
        for _, row in top3.iterrows()
    ]

    return {
        "theta_in": {
            "theta_ja": float(theta_ja_in),
            "theta_jb": float(theta_jb_in),
            "theta_jt": float(theta_jt_in),
        },
        "weights": {
            "wa": float(wa),
            "wb": float(wb),
            "wt": float(wt),
            "w1": float(w1),
            "w2": float(w2),
            "w3": float(w3),
            "w4": float(w4),
        },
        "all_points": all_points,
        "query_point": query_point,
        "neighbors": neighbor_points,
        "lines": lines,
        "top3_table": top3_table,
    }
