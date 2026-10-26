import json
import os

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Title
add_md("""# 🌸 Lily Flower Classification Project
### Deep Learning Assignment: Scratch CNN vs. Transfer Learning
**Guidelines Compliance:**
* **Self-collected Dataset**: Collected directly via Wikipedia/Wikimedia API (no Kaggle). Organized into two folders (`dataset/lily` and `dataset/other_flowers`).
* **Models**:
  1. Custom Convolutional Neural Network trained from scratch.
  2. MobileNetV2 with Transfer Learning (fine-tuned).
* **Interactive Testing**: Upload any photo from your computer/mobile to test real-time predictions.""")

# Setup
add_md("## Step 1: Check Environment & Hardware Acceleration (GPU/CPU)")
add_code("""import torch
import torchvision
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Active Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU. (Tip: You can enable free GPU in Colab via Runtime -> Change runtime type -> T4 GPU)")""")

# Step 2: Dataset Collection
add_md("""## Step 2: Automated Self-Collected Dataset (No Kaggle)
Fetches authentic, high-quality images directly into two folders:
* `dataset/lily` (Various Lily species)
* `dataset/other_flowers` (Roses, Sunflowers, Tulips, Orchids, Daisies, etc.)""")

add_code("""import os
import time
import requests
from PIL import Image
from io import BytesIO

BASE_DIR = "."
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

HEADERS = {
    "User-Agent": "LilyClassificationColab/1.0 (academic.research@uni.edu) requests/2.32"
}

CATEGORIES = {
    "lily": [
        "Lilium", "Lilium_candidum", "Lilium_lancifolium", "Lilium_bulbiferum",
        "Lilium_martagon", "Lilium_regale", "Lilium_superbum", "Lilium_auratum",
        "Lilium_speciosum", "Lilium_longiflorum", "Nymphaea_alba", "Spathiphyllum"
    ],
    "other_flowers": [
        "Rosa", "Rose", "Rosa_gallica", "Rosa_chinensis", "Helianthus_annuus",
        "Tulipa", "Tulipa_gesneriana", "Bellis_perennis", "Tagetes_erecta",
        "Taraxacum_officinale", "Hibiscus_syriacus", "Nelumbo_nucifera", "Dahlia"
    ]
}

BLACKLIST = ["icon", "logo", "flag", "map", "stub", "diagram", "symbol", "drawing", "wikimedia", "button"]

def fetch_image_urls(titles):
    urls = []
    for i in range(0, len(titles), 6):
        batch = titles[i:i+6]
        url = (
            f"https://en.wikipedia.org/w/api.php?action=query&generator=images"
            f"&titles={'|'.join(batch)}&gimlimit=250&prop=imageinfo&iiprop=url|mime&iiurlwidth=400&format=json"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    t = p.get("title", "").lower()
                    if any(b in t for b in BLACKLIST):
                        continue
                    ii = p.get("imageinfo", [])
                    if ii:
                        u = ii[0].get("thumburl", "") or ii[0].get("url", "")
                        if u and ("jpeg" in u or "jpg" in u or "png" in u):
                            urls.append(u)
        except Exception as e:
            pass
        time.sleep(0.2)
    return urls

def download_and_verify(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200: return False
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if img.width < 100 or img.height < 100: return False
        img.thumbnail((300, 300))
        img.save(path, "JPEG", quality=85)
        return True
    except:
        if os.path.exists(path): os.remove(path)
        return False

# Download images
os.makedirs(os.path.join(DATASET_DIR, "lily"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "other_flowers"), exist_ok=True)

for c_name, titles in CATEGORIES.items():
    folder = os.path.join(DATASET_DIR, c_name)
    existing = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(existing) >= 80:
        print(f"[{c_name}] Already have {len(existing)} images. Skipping.")
        continue
    print(f"Collecting images for [{c_name}]...")
    urls = fetch_image_urls(titles)
    count = len(existing)
    for u in urls:
        if count >= 80: break
        fname = f"{c_name}_{count+1:04d}.jpg"
        if download_and_verify(u, os.path.join(folder, fname)):
            count += 1
            if count % 20 == 0:
                print(f"  {c_name}: {count}/80 downloaded")
    print(f"Total verified {c_name} images: {count}")

print("\\nDataset ready in 'dataset/' directory!")""")

# Step 3: Data Loaders
add_md("## Step 3: Data Preprocessing, Augmentation & Train/Val Split (80/20)")
add_code("""import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = datasets.ImageFolder(DATASET_DIR)
class_names = full_dataset.classes
print(f"Classes: {class_names}")

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_loader = DataLoader(TransformedDataset(train_subset, data_transforms['train']), batch_size=16, shuffle=True)
val_loader = DataLoader(TransformedDataset(val_subset, data_transforms['val']), batch_size=16, shuffle=False)

print(f"Total: {len(full_dataset)} | Train: {train_size} | Validation: {val_size}")""")

# Step 4: Model Architectures
add_md("## Step 4: Define Both Architectures (Custom CNN & MobileNetV2)")
add_code("""import torch.nn as nn
from torchvision import models

# 1. Custom CNN from Scratch
class CustomLilyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomLilyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

# 2. Transfer Learning (MobileNetV2)
def build_transfer_model(num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model

print("Architectures successfully defined.")""")

# Step 5: Training Helper & Execution
add_md("## Step 5: Train Both Models")
add_code("""import copy
import torch.optim as optim

def train_network(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=12, name="Model"):
    print(f"\\n---> Training: {name} on {device}")
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        r_loss, r_corr = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            r_loss += loss.item() * inputs.size(0)
            r_corr += torch.sum(preds == labels.data).item()

        if scheduler: scheduler.step()
        t_loss = r_loss / len(train_loader.dataset)
        t_acc = r_corr / len(train_loader.dataset)

        model.eval()
        v_loss, v_corr = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                v_loss += loss.item() * inputs.size(0)
                v_corr += torch.sum(preds == labels.data).item()

        val_loss = v_loss / len(val_loader.dataset)
        val_acc = v_corr / len(val_loader.dataset)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Acc: {t_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Loss: {val_loss:.4f})")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

    print(f"Best Val Accuracy for {name}: {best_acc*100:.2f}%")
    model.load_state_dict(best_wts)
    return model, history

criterion = nn.CrossEntropyLoss()

# 1. Train Scratch CNN
scratch_model = CustomLilyCNN(num_classes=2).to(device)
opt_scratch = optim.Adam(scratch_model.parameters(), lr=0.0005, weight_decay=1e-4)
sch_scratch = optim.lr_scheduler.StepLR(opt_scratch, step_size=6, gamma=0.5)
scratch_model, scratch_history = train_network(
    scratch_model, train_loader, val_loader, criterion, opt_scratch, sch_scratch, num_epochs=12, name="Custom CNN (Scratch)"
)
torch.save(scratch_model.state_dict(), "custom_cnn_lily.pth")

# 2. Train Transfer Learning
transfer_model = build_transfer_model(num_classes=2).to(device)
opt_transfer = optim.Adam(transfer_model.classifier.parameters(), lr=0.001, weight_decay=1e-4)
sch_transfer = optim.lr_scheduler.StepLR(opt_transfer, step_size=5, gamma=0.5)
transfer_model, transfer_history = train_network(
    transfer_model, train_loader, val_loader, criterion, opt_transfer, sch_transfer, num_epochs=10, name="MobileNetV2 (Transfer)"
)
torch.save(transfer_model.state_dict(), "transfer_mobilenet_lily.pth")
print("\\nTraining complete! Both model weights saved.")""")

# Step 6: Visualizations & Evaluation
add_md("## Step 6: Comparison Plots & Detailed Evaluation Report")
add_code("""import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Plotting Loss & Accuracy curves
plt.figure(figsize=(13, 4.5))

plt.subplot(1, 2, 1)
plt.plot(scratch_history['val_loss'], label='Scratch CNN Val Loss', color='blue')
plt.plot(transfer_history['val_loss'], label='MobileNetV2 Val Loss', color='green')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(scratch_history['val_acc'], label='Scratch CNN Val Acc', color='blue')
plt.plot(transfer_history['val_acc'], label='MobileNetV2 Val Acc', color='green')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluation on Validation set
def show_report(model, loader, title):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_p.extend(preds.cpu().numpy())
            all_y.extend(y.numpy())
    print(f"\\n{'='*20} {title} {'='*20}")
    print(classification_report(all_y, all_p, target_names=class_names))

show_report(scratch_model, val_loader, "Custom CNN (From Scratch)")
show_report(transfer_model, val_loader, "MobileNetV2 (Transfer Learning)")""")

# Step 7: Interactive Upload & Test
add_md("""## Step 7: 🔍 Interactive Image Tester
**Click the button below to upload ANY image from your device (phone/PC) to test predictions in real-time!**""")

add_code("""from google.colab import files
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("Upload an image of a flower to test:")
uploaded = files.upload()

for filename in uploaded.keys():
    img_raw = Image.open(filename).convert("RGB")
    tensor = data_transforms['val'](img_raw).unsqueeze(0).to(device)

    # Scratch Model
    with torch.no_grad():
        s_probs = F.softmax(scratch_model(tensor), dim=1)[0]
        s_pred = class_names[torch.argmax(s_probs).item()]
        s_conf = s_probs[torch.argmax(s_probs).item()].item() * 100

        t_probs = F.softmax(transfer_model(tensor), dim=1)[0]
        t_pred = class_names[torch.argmax(t_probs).item()]
        t_conf = t_probs[torch.argmax(t_probs).item()].item() * 100

    # Display image & results
    plt.figure(figsize=(6, 6))
    plt.imshow(img_raw)
    plt.axis("off")
    title_text = (
        f"File: {filename}\\n"
        f"Custom CNN: {s_pred.upper()} ({s_conf:.1f}% conf)\\n"
        f"MobileNetV2: {t_pred.upper()} ({t_conf:.1f}% conf)"
    )
    plt.title(title_text, fontsize=13, fontweight='bold', pad=15)
    plt.show()

    print(f"Results for '{filename}':")
    print(f"  [Model 1 - Scratch CNN] : {s_pred} (Confidence: {s_conf:.2f}%)")
    print(f"  [Model 2 - MobileNetV2] : {t_pred} (Confidence: {t_conf:.2f}%)")
    print("-" * 55)""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

nb_path = "/Users/erbiumtech/.gemini/antigravity/scratch/lily_classification/Lily_Flower_Classification.ipynb"
with open(nb_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully created at: {nb_path}")
