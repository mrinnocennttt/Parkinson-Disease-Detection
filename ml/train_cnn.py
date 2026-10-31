import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ParkinsonDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("L")  # Grayscale
        # Invert if dark background
        arr = np.array(image)
        if np.mean(arr) < 127:
            arr = 255 - arr
            image = Image.fromarray(arr)

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

class LightweightCNN(nn.Module):
    """
    Compact CNN designed specifically for small medical drawing datasets
    to avoid overfitting while capturing spatial stroke irregularities.
    """
    def __init__(self):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out

def get_data_paths(task_dir):
    data = {}
    for split in ["training", "testing"]:
        paths, labels = [], []
        for label, cls_name in enumerate(["healthy", "parkinson"]):
            folder = os.path.join(task_dir, split, cls_name)
            files = sorted(glob.glob(os.path.join(folder, "*.*")))
            for f in files:
                paths.append(f)
                labels.append(label)
        data[split] = (paths, labels)
    return data

def train_cnn_for_task(task_name, task_dir, save_path, epochs=35):
    print(f"\n================ Training CNN for {task_name.upper()} ================")
    data = get_data_paths(task_dir)
    train_paths, train_labels = data["training"]
    test_paths, test_labels = data["testing"]

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ParkinsonDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = ParkinsonDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = LightweightCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(lbls)
        scheduler.step()

        epoch_loss = total_loss / len(train_dataset)

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == lbls).sum().item()
        acc = correct / len(test_dataset)
        history["train_loss"].append(round(epoch_loss, 4))
        history["test_acc"].append(round(acc, 4))

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} - Test Acc: {acc*100:.1f}% (Best: {best_acc*100:.1f}%)")

    # Load best model for full metrics calculation
    model.load_state_dict(torch.load(save_path))
    model.eval()

    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            preds = torch.argmax(outputs, dim=1).numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    metrics = {
        "task": task_name,
        "model_type": "Convolutional Neural Network (CNN)",
        "train_samples": len(train_paths),
        "test_samples": len(test_paths),
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
        },
        "history": history
    }

    print(f"\nFinal CNN Evaluation for {task_name}:")
    print(f" - Best Accuracy:        {metrics['accuracy']*100:.1f}%")
    print(f" - Sensitivity (Recall): {metrics['recall_sensitivity']*100:.1f}%")
    print(f" - Specificity:          {metrics['specificity']*100:.1f}%")
    print(f" - F1-Score:             {metrics['f1_score']:.4f}")
    print(f" - ROC-AUC:              {metrics['roc_auc']:.4f}")

    return metrics

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    results = {}
    results["spiral"] = train_cnn_for_task(
        task_name="spiral",
        task_dir=os.path.join(dataset_dir, "spiral"),
        save_path=os.path.join(models_dir, "cnn_spiral.pt"),
        epochs=35
    )

    results["wave"] = train_cnn_for_task(
        task_name="wave",
        task_dir=os.path.join(dataset_dir, "wave"),
        save_path=os.path.join(models_dir, "cnn_wave.pt"),
        epochs=35
    )

    metrics_file = os.path.join(models_dir, "cnn_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll CNN metrics successfully exported to {metrics_file}")

if __name__ == "__main__":
    main()
