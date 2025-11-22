// --- 전역 상태 ---
let conditionFilename = null;
// let materialFilename = null;

// DOM 로드 시점에 페이지 타입에 따라 초기화
window.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("run-btn");

  if (runBtn) {
    // input.html
    initSliders();
    initUploads();
    initRunButton();
  } else if (typeof RESULT !== "undefined") {
    // result.html
    renderPlot(RESULT);
    renderTop3(RESULT.top3_table);
  }
});

// --- 슬라이더 값 표시 ---
function initSliders() {
  ["k1", "k2", "k3", "k4"].forEach((id) => {
    const slider = document.getElementById(id);
    const label = document.getElementById(id + "-val");
    if (!slider || !label) return;
    const update = () => {
      label.textContent = slider.value + "%";
    };
    slider.addEventListener("input", update);
    update();
  });
}

// --- 파일 업로드(조건 / 재료) ---
function initUploads() {
  // 조건 파일
  const condInput = document.getElementById("condition-file");
  if (condInput) {
    condInput.addEventListener("change", async () => {
      const file = condInput.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload_condition", {
          method: "POST",
          body: formData,
        });
        const json = await res.json();
        if (json.ok) {
          conditionFilename = json.filename;
          alert("조건 파일 업로드 완료: " + json.filename);
        } else {
          alert("조건 파일 업로드 실패: " + json.error);
        }
      } catch (err) {
        console.error(err);
        alert("조건 파일 업로드 중 오류가 발생했습니다.");
      }
    });
  }

  // // (선택) 재료 DB — input.html에서 주석 처리했다면 없어도 OK
  // const matInput = document.getElementById("material-file");
  // if (matInput) {
  //   matInput.addEventListener("change", async () => {
  //     const file = matInput.files[0];
  //     if (!file) return;

  //     const formData = new FormData();
  //     formData.append("file", file);

  //     try {
  //       const res = await fetch("/api/upload_material", {
  //         method: "POST",
  //         body: formData,
  //       });
  //       const json = await res.json();
  //       if (json.ok) {
  //         materialFilename = json.filename;
  //         alert("재료 DB 파일 업로드 완료: " + json.filename);
  //       } else {
  //         alert("재료 DB 업로드 실패: " + json.error);
  //       }
  //     } catch (err) {
  //       console.error(err);
  //       alert("재료 DB 업로드 중 오류가 발생했습니다.");
  //     }
  //   });
  // }
}

// --- 분석 실행 버튼 ---
function initRunButton() {
  const btn = document.getElementById("run-btn");
  btn.addEventListener("click", async () => {
    if (!conditionFilename) {
      alert("조건 파일을 먼저 업로드해주세요.");
      return;
    }

    const k1 = document.getElementById("k1").value;
    const k2 = document.getElementById("k2").value;
    const k3 = document.getElementById("k3").value;
    const k4 = document.getElementById("k4").value;

    const ka = document.getElementById("ka").value || 1;
    const kb = document.getElementById("kb").value || 1;
    const kt = document.getElementById("kt").value || 1;

    const payload = {
      k1, k2, k3, k4,
      ka, kb, kt,
      conditionFilename,
      //materialFilename,
    };

    try {
      const res = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = await res.json();
      if (!json.ok) {
        alert("에러: " + json.error);
        return;
      }
      window.location.href = json.redirect;
    } catch (err) {
      console.error(err);
      alert("서버 호출 중 오류가 발생했습니다.");
    }
  });
}

// --- result 페이지: 3D 플롯 ---
function renderPlot(result) {
  const all = result.all_points;
  const query = result.query_point;
  const neighbors = result.neighbors;
  const lines = result.lines;

  const traceAll = {
    x: all.x,
    y: all.y,
    z: all.z,
    mode: "markers",
    type: "scatter3d",
    marker: { size: 3, color: "lightgray" },
    name: "All Materials",
  };

  const traceQuery = {
    x: [query.x],
    y: [query.y],
    z: [query.z],
    mode: "markers",
    type: "scatter3d",
    marker: { size: 8, color: "red", symbol: "star" },
    name: "Your Input",
  };

  const traceNeighbors = {
    x: neighbors.x,
    y: neighbors.y,
    z: neighbors.z,
    mode: "markers+text",
    type: "scatter3d",
    marker: { size: 6, color: "blue" },
    text: neighbors.names,
    textposition: "top center",
    name: "Top Neighbors",
  };

  const lineTraces = lines.map((l) => ({
    x: l.x,
    y: l.y,
    z: l.z,
    mode: "lines",
    type: "scatter3d",
    line: { color: "gray", width: 2, dash: "dot" },
    showlegend: false,
  }));

  const data = [traceAll, traceQuery, traceNeighbors, ...lineTraces];

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: "Theta_JA (normalized & weighted)" },
      yaxis: { title: "Theta_JB (normalized & weighted)" },
      zaxis: { title: "Theta_JT (normalized & weighted)" },
    },
  };

  const config = { responsive: true, scrollZoom: true };

  Plotly.newPlot("theta3d", data, layout, config);
}

// --- result 페이지: TOP3 테이블 ---
function renderTop3(top3) {
  const tbody = document.querySelector("#top3-table tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  top3.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.lank}</td>
      <td>${row.name}</td>
      <td>${row.score}</td>
    `;
    tbody.appendChild(tr);
  });
}
