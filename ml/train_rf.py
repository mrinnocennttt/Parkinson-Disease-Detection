import os
import glob
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from features import preprocess_image, extract_combined_features

def load_data(task_dir):
    """
    Load images and extract hybrid features for training and testing sets.
    Directory structure:
      task_dir/training/healthy
      task_dir/training/parkinson
      task_dir/testing/healthy
      task_dir/testing/parkinson
    """
    sets = {}
    for split in ["training", "testing"]:
        X, y, paths = [], [], []
        for label, cls_name in enumerate(["healthy", "parkinson"]):
            folder = os.path.join(task_dir, split, cls_name)
            img_files = sorted(glob.glob(os.path.join(folder, "*.*")))
            for p in img_files:
                try:
                    img = preprocess_image(p)
                    feat, _ = extract_combined_features(img)
                    X.append(feat)
                    y.append(label)
                    paths.append(p)
                except Exception as e:
                    print(f"Error processing {p}: {e}")
        sets[split] = (np.array(X, dtype=np.float32), np.array(y, dtype=int), paths)
    return sets

def train_and_evaluate(task_name, task_dir, model_save_path):
    print(f"\n================ Training Random Forest for {task_name.upper()} ================")
    data = load_data(task_dir)
    X_train, y_train, train_paths = data["training"]
    X_test, y_test, test_paths = data["testing"]

    print(f"Train samples: {len(y_train)} (Healthy: {sum(y_train==0)}, PD: {sum(y_train==1)})")
    print(f"Test samples:  {len(y_test)} (Healthy: {sum(y_test==0)}, PD: {sum(y_test==1)})")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")

    # Optimize Random Forest hyperparameters
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Test evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "task": task_name,
        "model_type": "Random Forest (HOG + Biomechanical Biomarkers)",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr]
        }
    }

    print(f"Results for {task_name}:")
    print(f" - Accuracy:            {metrics['accuracy']*100:.1f}%")
    print(f" - Sensitivity (Recall): {metrics['recall_sensitivity']*100:.1f}%")
    print(f" - Specificity:         {metrics['specificity']*100:.1f}%")
    print(f" - F1-Score:            {metrics['f1_score']:.4f}")
    print(f" - ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f" - Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf, model_save_path)
    print(f"Model saved to {model_save_path}")

    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    results = {}
    
    # Spiral Test
    spiral_metrics = train_and_evaluate(
        task_name="spiral",
        task_dir=os.path.join(dataset_dir, "spiral"),
        model_save_path=os.path.join(models_dir, "rf_spiral.joblib")
    )
    results["spiral"] = spiral_metrics

    # Wave Test
    wave_metrics = train_and_evaluate(
        task_name="wave",
        task_dir=os.path.join(dataset_dir, "wave"),
        model_save_path=os.path.join(models_dir, "rf_wave.joblib")
    )
    results["wave"] = wave_metrics

    metrics_file = os.path.join(models_dir, "rf_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Random Forest metrics successfully exported to {metrics_file}")

if __name__ == "__main__":
    main()
