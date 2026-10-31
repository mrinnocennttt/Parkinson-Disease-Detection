# NeuroDraw AI — Parkinson’s Disease Detection via Spiral & Wave Drawings

An end-to-end clinical machine learning and computer vision healthcare application designed to detect early-stage **Parkinson’s Disease (PD)** through patient hand-drawn motor tasks (Archimedean Spirals and Sine Waves).

---

## 🔬 Clinical Background & Rationale
Parkinson's Disease is characterized by dopaminergic neuron degeneration within the substantia nigra, impairing basal ganglia motor loop regulation. This leads to **resting and kinetic tremors (4–7 Hz), bradykinesia, rigidity, and dysmetria**.
When asked to draw continuous geometric shapes (spirals and sine waves), healthy individuals maintain smooth trajectory continuity and uniform curvature, whereas patients with early-stage Parkinson's exhibit involuntary micro-oscillations, angular hesitations, and jagged line deviations.

---

## 🚀 Key Features

1. **Dual Machine Learning Architectures**:
   - **Deep Convolutional Neural Network (PyTorch CNN)**: Custom 4-block CNN with Batch Normalization, LeakyReLU, and Dropout optimized for medical drawing classification.
     - **Spiral Accuracy**: **83.3%** (93.3% Sensitivity / Recall)
     - **Wave Accuracy**: **100.0%** (100% Sensitivity, 100% Specificity)
   - **Hybrid Tabular Random Forest (Scikit-Learn)**: 150-estimator balanced forest utilizing Histogram of Oriented Gradients (HOG) combined with biomechanical tremor biomarkers.

2. **Computer Vision & Tremor Biomarker Extraction**:
   - **Tremor Severity Index**: Composite 0–100 score quantifying motor tremor roughness.
   - **Curvature Variance & Jaggedness Ratio**: Identifies micro-wobbles and sharp angular deviations.
   - **2D FFT Spectral Energy Ratio**: Isolates high-frequency spectral oscillations typical of kinetic tremors.
   - **Gradient Directional Entropy**: Captures disorganized stroke trajectories.
   - **Tremor Heatmap Overlay**: Real-time JET colormap overlay localizing points of high-frequency tremor directly on the drawing.

3. **Interactive Healthcare Web Application**:
   - **Live Drawing Canvas**: Interactive HTML5 touch/mouse pad with reference ghost guides (Archimedean spiral and sine wave).
   - **Scanned Drawing Upload**: Drag-and-drop support for clinical scans (PNG, JPG, TIFF).
   - **Benchmark Test Library**: 1-click evaluation using verified Kaggle benchmark patient samples.
   - **Clinical Decision Support**: Provides risk probability scores, UPDRS / Hoehn & Yahr clinical stage estimates, and actionable clinical next-step recommendations.
   - **Printable Clinical Summary Report**: Formatted clinical summary ready for export or printing.
   - **Model Intelligence Hub**: Interactive confusion matrices, ROC metrics, and comparative evaluation dashboard.

---

## 📂 Project Structure

```
├── app.py                  # Flask backend REST API and diagnostic inference engine
├── dataset/                # Kaggle Parkinson's Drawing Dataset
│   ├── spiral/             # Archimedean Spiral drawings (training/testing)
│   └── wave/               # Sine Wave drawings (training/testing)
├── ml/
│   ├── features.py         # HOG, Biomechanical tremor analysis & heatmap generator
│   ├── train_rf.py         # Random Forest training and evaluation pipeline
│   └── train_cnn.py        # PyTorch CNN training, data augmentation & evaluation
├── models/
│   ├── cnn_spiral.pt       # PyTorch trained weights (Spiral)
│   ├── cnn_wave.pt         # PyTorch trained weights (Wave)
│   ├── rf_spiral.joblib    # Trained Random Forest (Spiral)
│   ├── rf_wave.joblib      # Trained Random Forest (Wave)
│   ├── cnn_metrics.json    # Evaluated benchmark test metrics (CNN)
│   └── rf_metrics.json     # Evaluated benchmark test metrics (RF)
├── static/
│   ├── css/style.css       # Clinical dark-mode design system & animations
│   └── js/app.js           # Live canvas engine, API interaction & analytics rendering
└── templates/
    └── index.html          # Responsive healthcare application UI
```

---

## 🛠️ Quickstart Guide

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Retrain Models (Optional)
```bash
# Train Random Forest models
PYTHONPATH=. python ml/train_rf.py

# Train PyTorch CNN models
PYTHONPATH=. python ml/train_cnn.py
```

### 3. Launch Healthcare Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5001`**
