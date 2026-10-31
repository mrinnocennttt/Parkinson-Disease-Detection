// NeuroDraw AI — Frontend Application Logic
document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentTask = "spiral";       // "spiral" or "wave"
  let currentModel = "cnn";         // "cnn" or "rf"
  let currentInputMode = "draw";    // "draw", "upload", or "samples"
  let isDrawing = false;
  let strokeHistory = [];
  let currentStroke = [];
  let uploadedFile = null;
  let selectedSampleUrl = null;
  let showGhostGuide = true;
  let lastDiagnosisResult = null;

  // DOM Elements
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");
  const ghostGuideCanvas = document.getElementById("ghostGuideCanvas");
  const ghostCtx = ghostGuideCanvas.getContext("2d");

  const btnAnalyze = document.getElementById("btnAnalyze");
  const btnClear = document.getElementById("btnClear");
  const btnUndo = document.getElementById("btnUndo");
  const btnToggleGuide = document.getElementById("btnToggleGuide");
  const strokeWidthInput = document.getElementById("strokeWidth");

  const resultsCard = document.getElementById("resultsCard");
  const resultsEmpty = document.getElementById("resultsEmpty");
  const resultsContent = document.getElementById("resultsContent");

  // Tabs & Views
  const navStudioBtn = document.getElementById("navStudio");
  const navMetricsBtn = document.getElementById("navMetrics");
  const studioView = document.getElementById("studioView");
  const metricsView = document.getElementById("metricsView");

  // Modality & Model Toggles
  const taskOptions = document.querySelectorAll("[data-task]");
  const modelOptions = document.querySelectorAll("[data-model]");
  const subTabBtns = document.querySelectorAll(".sub-tab-btn");

  // Panels
  const canvasSection = document.getElementById("canvasSection");
  const uploadSection = document.getElementById("uploadSection");
  const samplesSection = document.getElementById("samplesSection");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const uploadPreviewWrap = document.getElementById("uploadPreviewWrap");
  const uploadPreviewImg = document.getElementById("uploadPreviewImg");
  const samplesGrid = document.getElementById("samplesGrid");

  // --- Canvas Setup ---
  function initCanvases() {
    const size = 440;
    canvas.width = size;
    canvas.height = size;
    ghostGuideCanvas.width = size;
    ghostGuideCanvas.height = size;

    clearCanvas(false);
    drawGhostGuide();
  }

  function clearCanvas(recordHistory = true) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (recordHistory) {
      strokeHistory = [];
      currentStroke = [];
    }
  }

  function isCanvasBlank() {
    return strokeHistory.length === 0;
  }

  function drawGhostGuide() {
    ghostCtx.clearRect(0, 0, ghostGuideCanvas.width, ghostGuideCanvas.height);
    if (!showGhostGuide) return;

    ghostCtx.lineWidth = 2.5;
    ghostCtx.strokeStyle = "rgba(6, 182, 212, 0.45)";
    ghostCtx.setLineDash([6, 6]);

    const cx = ghostGuideCanvas.width / 2;
    const cy = ghostGuideCanvas.height / 2;

    if (currentTask === "spiral") {
      ghostCtx.beginPath();
      const a = 5;
      const b = 5.2;
      const maxTheta = 7.5 * Math.PI;
      for (let theta = 0; theta < maxTheta; theta += 0.05) {
        const r = a + b * theta;
        const x = cx + r * Math.cos(theta);
        const y = cy + r * Math.sin(theta);
        if (theta === 0) ghostCtx.moveTo(x, y);
        else ghostCtx.lineTo(x, y);
      }
      ghostCtx.stroke();
    } else {
      ghostCtx.beginPath();
      const width = ghostGuideCanvas.width;
      const height = ghostGuideCanvas.height;
      const padding = 35;
      const waveY = height / 2;
      const amplitude = 65;
      const frequency = 3.5;

      for (let x = padding; x <= width - padding; x += 3) {
        const progress = (x - padding) / (width - 2 * padding);
        const y = waveY + amplitude * Math.sin(progress * 2 * Math.PI * frequency);
        if (x === padding) ghostCtx.moveTo(x, y);
        else ghostCtx.lineTo(x, y);
      }
      ghostCtx.stroke();
    }
    ghostCtx.setLineDash([]);
  }

  // --- Drawing Events ---
  function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getCoordinates(e);
    currentStroke = [{ x, y }];

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = parseInt(strokeWidthInput.value, 10);
  }

  function drawMove(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getCoordinates(e);
    currentStroke.push({ x, y });

    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    e.preventDefault();
    isDrawing = false;
    ctx.closePath();
    if (currentStroke.length > 0) {
      strokeHistory.push({
        width: parseInt(strokeWidthInput.value, 10),
        points: currentStroke
      });
      currentStroke = [];
    }
  }

  function redrawAllStrokes() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";

    for (const stroke of strokeHistory) {
      if (stroke.points.length === 0) continue;
      ctx.lineWidth = stroke.width;
      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) {
        ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
      ctx.stroke();
      ctx.closePath();
    }
  }

  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", drawMove);
  window.addEventListener("mouseup", stopDrawing);

  canvas.addEventListener("touchstart", startDrawing, { passive: false });
  canvas.addEventListener("touchmove", drawMove, { passive: false });
  window.addEventListener("touchend", stopDrawing, { passive: false });

  btnClear.addEventListener("click", () => clearCanvas(true));
  btnUndo.addEventListener("click", () => {
    if (strokeHistory.length > 0) {
      strokeHistory.pop();
      redrawAllStrokes();
    }
  });

  btnToggleGuide.addEventListener("click", () => {
    showGhostGuide = !showGhostGuide;
    btnToggleGuide.classList.toggle("active-tool", showGhostGuide);
    drawGhostGuide();
  });

  // --- Task & Model Switchers ---
  taskOptions.forEach(btn => {
    btn.addEventListener("click", () => {
      taskOptions.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentTask = btn.dataset.task;
      drawGhostGuide();
      filterSamples();
    });
  });

  modelOptions.forEach(btn => {
    btn.addEventListener("click", () => {
      modelOptions.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentModel = btn.dataset.model;
    });
  });

  // --- Switch Input Mode Helper ---
  function switchInputMode(mode) {
    currentInputMode = mode;
    subTabBtns.forEach(b => {
      b.classList.toggle("active", b.dataset.input === mode);
    });

    canvasSection.style.display = mode === "draw" ? "block" : "none";
    uploadSection.style.display = mode === "upload" ? "block" : "none";
    samplesSection.style.display = mode === "samples" ? "block" : "none";

    if (mode === "samples") {
      loadSamples();
    }
  }

  subTabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      switchInputMode(btn.dataset.input);
    });
  });

  // --- File Upload / Drag-and-Drop ---
  dropzone.addEventListener("click", (e) => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    uploadedFile = file;
    selectedSampleUrl = null;
    switchInputMode("upload");

    const reader = new FileReader();
    reader.onload = (event) => {
      uploadPreviewImg.src = event.target.result;
      uploadPreviewWrap.style.display = "block";
    };
    reader.readAsDataURL(file);
  }

  // --- Samples Gallery ---
  let allSamples = [];
  async function loadSamples() {
    if (allSamples.length > 0) {
      filterSamples();
      return;
    }
    try {
      const resp = await fetch("/api/samples");
      const data = await resp.json();
      if (data.status === "success") {
        allSamples = data.samples;
        filterSamples();
      }
    } catch (err) {
      console.error("Error loading samples:", err);
    }
  }

  function filterSamples() {
    samplesGrid.innerHTML = "";
    const filtered = allSamples.filter(s => s.task === currentTask);
    filtered.forEach(sample => {
      const card = document.createElement("div");
      card.className = "sample-card";
      card.innerHTML = `
        <img src="${sample.url}" alt="${sample.label}">
        <span class="sample-badge ${sample.category === 'healthy' ? 'badge-healthy-sample' : 'badge-pd-sample'}">
          ${sample.category === 'healthy' ? 'Healthy Control' : 'Parkinson (PD)'}
        </span>
      `;
      card.addEventListener("click", () => {
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("selected"));
        card.classList.add("selected");
        selectedSampleUrl = sample.url;
        uploadedFile = null;
      });
      samplesGrid.appendChild(card);
    });
  }

  // --- Diagnostic Inference Execution ---
  btnAnalyze.addEventListener("click", runDiagnosis);

  async function runDiagnosis() {
    btnAnalyze.disabled = true;
    const originalText = btnAnalyze.innerHTML;
    btnAnalyze.innerHTML = `<span class="spinner"></span> Running Diagnostic Engine...`;

    try {
      const formData = new FormData();
      formData.append("task", currentTask);
      formData.append("model_type", currentModel);

      // Determine active input payload
      if (currentInputMode === "upload") {
        if (!uploadedFile) {
          alert("Please select or drop an image file first.");
          btnAnalyze.disabled = false;
          btnAnalyze.innerHTML = originalText;
          return;
        }
        formData.append("file", uploadedFile, uploadedFile.name);
      } else if (currentInputMode === "samples") {
        if (!selectedSampleUrl) {
          alert("Please select a sample drawing from the gallery.");
          btnAnalyze.disabled = false;
          btnAnalyze.innerHTML = originalText;
          return;
        }
        const sampleResp = await fetch(selectedSampleUrl);
        const sampleBlob = await sampleResp.blob();
        formData.append("file", sampleBlob, "sample.png");
      } else {
        // Draw Mode
        if (isCanvasBlank()) {
          alert("Please draw a spiral or wave on the canvas first, or switch to 'Upload File' or 'Test Library'.");
          btnAnalyze.disabled = false;
          btnAnalyze.innerHTML = originalText;
          return;
        }
        const dataUrl = canvas.toDataURL("image/png");
        formData.append("image_data", dataUrl);
      }

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const result = await response.json();
      if (result.status === "success") {
        lastDiagnosisResult = result;
        renderResults(result);
      } else {
        alert("Diagnostic Notification: " + (result.message || "An unexpected issue occurred."));
      }
    } catch (err) {
      console.error("Diagnosis error:", err);
      alert("Error contacting diagnostic server. Please make sure the app backend is active.");
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = originalText;
    }
  }

  // --- Results Rendering ---
  function renderResults(res) {
    resultsEmpty.style.display = "none";
    resultsContent.style.display = "flex";

    const banner = document.getElementById("diagBanner");
    const headline = document.getElementById("diagHeadline");
    const confVal = document.getElementById("diagConfVal");
    const statusPill = document.getElementById("diagStatusPill");
    const fillHealthy = document.getElementById("probHealthyFill");
    const fillPd = document.getElementById("probPdFill");
    const valHealthy = document.getElementById("probHealthyVal");
    const valPd = document.getElementById("probPdVal");

    const imgOrig = document.getElementById("resImgOrig");
    const imgHeatmap = document.getElementById("resImgHeatmap");

    const valTremor = document.getElementById("bioTremorIndex");
    const valCurv = document.getElementById("bioCurvVar");
    const valJag = document.getElementById("bioJaggedness");
    const valSpectral = document.getElementById("bioSpectralRatio");

    const clinicalStage = document.getElementById("clinicalStage");
    const clinicalRec = document.getElementById("clinicalRec");

    if (res.is_parkinson) {
      banner.className = "diagnosis-banner banner-parkinson";
      statusPill.textContent = "Neuromotor Signature Anomaly Detected";
      headline.textContent = "Elevated Parkinson's Risk";
    } else {
      banner.className = "diagnosis-banner banner-healthy";
      statusPill.textContent = "Physiological Line Fluidity Verified";
      headline.textContent = "Consistent with Healthy Control";
    }

    confVal.textContent = `${res.confidence_pct}%`;
    valHealthy.textContent = `${res.probability_healthy}%`;
    valPd.textContent = `${res.probability_parkinson}%`;

    fillHealthy.style.width = `${res.probability_healthy}%`;
    fillPd.style.width = `${res.probability_parkinson}%`;

    imgOrig.src = res.visuals.preprocessed_image;
    imgHeatmap.src = res.visuals.tremor_heatmap;

    valTremor.textContent = res.biomarkers.tremor_index;
    valCurv.textContent = Number(res.biomarkers.curvature_variance).toFixed(4);
    valJag.textContent = (Number(res.biomarkers.jaggedness_ratio) * 100).toFixed(1) + "%";
    valSpectral.textContent = Number(res.biomarkers.spectral_energy_ratio).toFixed(4);

    clinicalStage.textContent = `${res.clinical_assessment.stage} • ${res.clinical_assessment.severity}`;
    clinicalRec.textContent = res.clinical_assessment.recommendation;

    if (window.innerWidth < 1024) {
      resultsCard.scrollIntoView({ behavior: "smooth" });
    }
  }

  // --- Print / Export Clinical Report ---
  document.getElementById("btnExportReport").addEventListener("click", () => {
    if (!lastDiagnosisResult) return;
    window.print();
  });

  // --- Navigation & Analytics Tab ---
  navStudioBtn.addEventListener("click", () => {
    navStudioBtn.classList.add("active");
    navMetricsBtn.classList.remove("active");
    studioView.classList.add("active-view");
    metricsView.classList.remove("active-view");
  });

  navMetricsBtn.addEventListener("click", () => {
    navMetricsBtn.classList.add("active");
    navStudioBtn.classList.remove("active");
    metricsView.classList.add("active-view");
    studioView.classList.remove("active-view");
    loadModelMetrics();
  });

  async function loadModelMetrics() {
    try {
      const resp = await fetch("/api/metrics");
      const data = await resp.json();
      if (data.status === "success") {
        renderMetricsDashboard(data);
      }
    } catch (err) {
      console.error("Error loading metrics:", err);
    }
  }

  function renderMetricsDashboard(data) {
    const cnnSpiral = data.cnn?.spiral || {};
    const cnnWave = data.cnn?.wave || {};
    const rfSpiral = data.random_forest?.spiral || {};
    const rfWave = data.random_forest?.wave || {};

    document.getElementById("dashCnnSpiralAcc").textContent = (cnnSpiral.accuracy ? (cnnSpiral.accuracy * 100).toFixed(1) : "--") + "%";
    document.getElementById("dashCnnWaveAcc").textContent = (cnnWave.accuracy ? (cnnWave.accuracy * 100).toFixed(1) : "--") + "%";
    document.getElementById("dashRfSpiralAcc").textContent = (rfSpiral.accuracy ? (rfSpiral.accuracy * 100).toFixed(1) : "--") + "%";
    document.getElementById("dashRfWaveAcc").textContent = (rfWave.accuracy ? (rfWave.accuracy * 100).toFixed(1) : "--") + "%";

    if (cnnSpiral.confusion_matrix) {
      const cm = cnnSpiral.confusion_matrix;
      document.getElementById("cmSpiralTN").textContent = cm.tn;
      document.getElementById("cmSpiralFP").textContent = cm.fp;
      document.getElementById("cmSpiralFN").textContent = cm.fn;
      document.getElementById("cmSpiralTP").textContent = cm.tp;
    }

    if (cnnWave.confusion_matrix) {
      const cm = cnnWave.confusion_matrix;
      document.getElementById("cmWaveTN").textContent = cm.tn;
      document.getElementById("cmWaveFP").textContent = cm.fp;
      document.getElementById("cmWaveFN").textContent = cm.fn;
      document.getElementById("cmWaveTP").textContent = cm.tp;
    }

    const tableBody = document.getElementById("metricsTableBody");
    tableBody.innerHTML = `
      <tr>
        <td><strong>Lightweight CNN</strong></td>
        <td>Archimedean Spiral</td>
        <td><span style="color:var(--accent-healthy); font-weight:700;">${(cnnSpiral.accuracy * 100).toFixed(1)}%</span></td>
        <td>${(cnnSpiral.recall_sensitivity * 100).toFixed(1)}%</td>
        <td>${(cnnSpiral.specificity * 100).toFixed(1)}%</td>
        <td>${cnnSpiral.f1_score.toFixed(4)}</td>
        <td>${cnnSpiral.roc_auc.toFixed(4)}</td>
      </tr>
      <tr>
        <td><strong>Lightweight CNN</strong></td>
        <td>Sine Wave</td>
        <td><span style="color:var(--accent-healthy); font-weight:700;">${(cnnWave.accuracy * 100).toFixed(1)}%</span></td>
        <td>${(cnnWave.recall_sensitivity * 100).toFixed(1)}%</td>
        <td>${(cnnWave.specificity * 100).toFixed(1)}%</td>
        <td>${cnnWave.f1_score.toFixed(4)}</td>
        <td>${cnnWave.roc_auc.toFixed(4)}</td>
      </tr>
      <tr>
        <td><strong>Random Forest + HOG</strong></td>
        <td>Archimedean Spiral</td>
        <td>${(rfSpiral.accuracy * 100).toFixed(1)}%</td>
        <td>${(rfSpiral.recall_sensitivity * 100).toFixed(1)}%</td>
        <td>${(rfSpiral.specificity * 100).toFixed(1)}%</td>
        <td>${rfSpiral.f1_score.toFixed(4)}</td>
        <td>${rfSpiral.roc_auc.toFixed(4)}</td>
      </tr>
      <tr>
        <td><strong>Random Forest + HOG</strong></td>
        <td>Sine Wave</td>
        <td>${(rfWave.accuracy * 100).toFixed(1)}%</td>
        <td>${(rfWave.recall_sensitivity * 100).toFixed(1)}%</td>
        <td>${(rfWave.specificity * 100).toFixed(1)}%</td>
        <td>${rfWave.f1_score.toFixed(4)}</td>
        <td>${rfWave.roc_auc.toFixed(4)}</td>
      </tr>
    `;
  }

  initCanvases();
});
