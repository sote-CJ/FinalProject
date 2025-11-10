# plots/plot_builder.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 서버 환경용 백엔드
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.gridspec import GridSpec

def min_max_norm(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / max(x_max - x_min, eps)

def min_max_norm_invert(x, eps=1e-9):
    return 1.0 - min_max_norm(x, eps)

def spread_for_plot(x):
    x = np.asarray(x, dtype=float)
    return 0.8 * (x - np.min(x) + 0.1) / (np.max(x) - np.min(x) + 1e-9)

def build_figure(
    Ta, Tb, Tc, Tj,
    ka, kb, kt,
    k1, k2, k3, k4,
    excel_path
):
    P = 1.0  # 소비전력 1W 고정

    # 1) 가중치 정규화
    sum_kt = ka + kb + kt
    wa, wb, wt = ka / sum_kt, kb / sum_kt, kt / sum_kt

    sum_ke = k1 + k2 + k3 + k4
    w1, w2, w3, w4 = k1 / sum_ke, k2 / sum_ke, k3 / sum_ke, k4 / sum_ke

    # 2) theta 계산
    theta_ja_in = (Tj - Ta) / P
    theta_jb_in = (Tj - Tb) / P
    theta_jt_in = (Tj - Tc) / P

    # 3) 엑셀 로드
    df = pd.read_excel(excel_path)

    # 4) KNN 전처리
    df_knn = df.dropna(subset=["theta_ja", "theta_jb", "theta_jt"]).copy()
    if len(df_knn) == 0:
        raise ValueError("theta_ja, theta_jb, theta_jt 값이 모두 있는 행이 없습니다.")

    X = df_knn[["theta_ja", "theta_jb", "theta_jt"]].to_numpy()
    X_min = X.min(axis=0); X_max = X.max(axis=0)
    range_ = np.maximum(X_max - X_min, 1e-9)

    X_norm = (X - X_min) / range_
    Y = np.array([theta_ja_in, theta_jb_in, theta_jt_in])
    Y_norm = (Y - X_min) / range_

    # 디버그용 컬럼(선택)
    df_knn["theta_ja_norm"] = X_norm[:, 0]
    df_knn["theta_jb_norm"] = X_norm[:, 1]
    df_knn["theta_jt_norm"] = X_norm[:, 2]

    # 5) 가중치 적용 후 KNN
    w = np.array([wa, wb, wt])
    sqrt_w = np.sqrt(w)
    X_w = X_norm * sqrt_w
    Y_w = Y_norm * sqrt_w

    n_neighbors = min(5, len(X_w))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(X_w)
    distances, indices = nbrs.kneighbors(Y_w.reshape(1, -1))

    top_idx = indices[0]
    df_top5 = df_knn.iloc[top_idx].copy()
    df_top5["distance"] = distances.flatten()

    # 6) ECO SCORE
    theta_ja_mat = df_top5["theta_ja"].to_numpy()
    co2_ratio = theta_ja_in / np.maximum(theta_ja_mat, 1e-9)
    df_top5["score_co2"] = min_max_norm(co2_ratio)

    tg = df_top5["Tg"].to_numpy()
    cte = df_top5["cte"].to_numpy()
    score_life = 0.7 * min_max_norm(tg) + 0.3 * min_max_norm_invert(cte)
    df_top5["score_life"] = score_life

    recycle_map = {"A": 1.0, "B": 0.67, "C": 0.33, "D": 0.0}
    df_top5["score_recycle"] = df_top5["recycle"].map(recycle_map).fillna(0.0)

    df_top5["score_cost"] = 0.0  # TODO: 실제 가격 지표 반영

    df_top5["score_eco"] = (
        w1 * df_top5["score_co2"] +
        w2 * df_top5["score_life"] +
        w3 * df_top5["score_recycle"] +
        w4 * df_top5["score_cost"]
    )

    # 7) 최종 점수
    df_top5["score_thermal"] = min_max_norm_invert(df_top5["distance"].to_numpy())
    df_top5["score_final"] = 0.5 * df_top5["score_thermal"] + 0.5 * df_top5["score_eco"]
    df_top5_ranked = df_top5.sort_values(by="score_final", ascending=False).reset_index(drop=True)

    # 8) Figure/axes
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, width_ratios=[1.3, 1.1, 0.8], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # ax1: 3D 산점도
    ax1.scatter(X_w[:, 0], X_w[:, 1], X_w[:, 2], c="lightgray", label="All Materials")
    ax1.scatter(Y_w[0], Y_w[1], Y_w[2], c="red", marker="*", s=200, label="Your Input")
    ax1.scatter(X_w[top_idx, 0], X_w[top_idx, 1], X_w[top_idx, 2], c="blue", s=100, edgecolors="black",
                label=f"Top {n_neighbors} Neighbors")
    for i in top_idx:
        ax1.plot([Y_w[0], X_w[i, 0]], [Y_w[1], X_w[i, 1]], [Y_w[2], X_w[i, 2]],
                 c="gray", linestyle="dotted", linewidth=1)

    ymin, ymax = ax1.get_ylim(); y_range = ymax - ymin; base_offset = 0.02 * y_range
    offsets = np.linspace(-1, 1, len(top_idx)) * base_offset
    for (i, name, dy) in zip(top_idx, df_top5["name"], offsets):
        x, y, z = X_w[i, 0], X_w[i, 1], X_w[i, 2]
        ax1.text(x, y + dy, z, str(name), color="blue", fontsize=9, ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    ax1.set_xlabel("Theta_JA (normalized & weighted)")
    ax1.set_ylabel("Theta_JB (normalized & weighted)")
    ax1.set_zlabel("Theta_JT (normalized & weighted)")
    ax1.set_title("3D Theta Space (Weighted KNN)")
    ax1.legend(loc="upper left")

    # ax2: Thermal vs Eco 막대
    df_bar = df_top5_ranked.sort_values(by="score_thermal", ascending=False).reset_index(drop=True)
    names = df_bar["name"].to_numpy()
    thermal_scores = df_bar["score_thermal"].to_numpy()
    eco_scores = df_bar["score_eco"].to_numpy()
    thermal_plot = spread_for_plot(thermal_scores)
    eco_plot = spread_for_plot(eco_scores)

    y_pos = np.arange(len(df_bar)); bar_h = 0.35
    ax2.barh(y_pos - bar_h/2, thermal_plot, height=bar_h, label="Thermal", alpha=0.8)
    ax2.barh(y_pos + bar_h/2, eco_plot, height=bar_h, label="Eco", alpha=0.8)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names)
    ax2.set_xlim(0, 1); ax2.invert_yaxis()
    ax2.set_xlabel("Score (0–1)"); ax2.set_title("Thermal vs Eco Scores (Top 5)")
    for i, (tp, ep, ts, es) in enumerate(zip(thermal_plot, eco_plot, thermal_scores, eco_scores)):
        ax2.text(tp + 0.02, y_pos[i] - bar_h/2, f"{ts:.2f}", va="center", fontsize=8)
        ax2.text(ep + 0.02, y_pos[i] + bar_h/2, f"{es:.2f}", va="center", fontsize=8)
    ax2.legend(loc="lower right"); ax2.grid(axis="x", linestyle="--", alpha=0.3)

    # ax3: Top 3 표
    ax3.axis("off")
    top3 = df_top5_ranked.head(3).copy()
    top3["LANK"] = np.arange(1, len(top3) + 1)
    top3["SCORE"] = (top3["score_final"] * 100).round().astype(int)
    table_data = top3[["LANK", "name", "SCORE"]].values.tolist()
    col_labels = ["LANK", "NAME", "SCORE"]
    table = ax3.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 2.0)
    ax3.set_title("Top 3 Materials (Final Score)", pad=20)

    fig.tight_layout()
    return fig
