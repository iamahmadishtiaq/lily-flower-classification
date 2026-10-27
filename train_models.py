import os
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Custom CNN Architecture (Trained from scratch)
class CustomLilyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomLilyCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28

            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4 output
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. Transfer Learning Model (MobileNetV2)
def build_transfer_model(num_classes=3):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze base feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace the final classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model

# 3. Training Loop Helper
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, device="cpu", model_name="model"):
    print(f"\n---> Starting training for: {model_name} on {device}")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        if scheduler:
            scheduler.step()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data).item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = running_val_corrects / len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Best Val Acc: {best_acc*100:.2f}%")

    model.load_state_dict(best_model_wts)
    return model, history

# 4. Evaluation Helper
def evaluate_model(model, val_loader, class_names, device="cpu", model_title="Model"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n--- Evaluation Report for {model_title} ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return report, cm

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using computing device: {device}")

    # Data transforms
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

    # Load Full Dataset
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    class_names = full_dataset.classes
    print(f"Detected classes ({len(class_names)}): {class_names}")

    # Split train/val 80/20 with fixed seed for fair comparison
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return self.transform(x), y
        def __len__(self):
            return len(self.subset)

    train_data = TransformedSubset(train_subset, data_transforms['train'])
    val_data = TransformedSubset(val_subset, data_transforms['val'])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)

    print(f"Total samples: {len(full_dataset)} | Training samples: {train_size} | Validation samples: {val_size}")

    # ================= 1. Train Scratch CNN =================
    print("\n" + "="*50)
    print(">>> 1. Training Custom CNN from Scratch <<<")
    print("="*50)
    scratch_model = CustomLilyCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_scratch = optim.Adam(scratch_model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler_scratch = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=7, gamma=0.5)

    scratch_model, scratch_history = train_model(
        scratch_model, train_loader, val_loader, criterion, optimizer_scratch,
        scheduler_scratch, num_epochs=15, device=device, model_name="Custom CNN (Scratch)"
    )
    scratch_report, scratch_cm = evaluate_model(scratch_model, val_loader, class_names, device, "Custom CNN (Scratch)")
    
    # Save scratch model weights
    scratch_save_path = os.path.join(OUTPUT_DIR, "custom_cnn_lily.pth")
    torch.save(scratch_model.state_dict(), scratch_save_path)
    print(f"Saved Custom CNN weights to {scratch_save_path}")

    # ================= 2. Train Transfer Learning (MobileNetV2) =================
    print("\n" + "="*50)
    print(">>> 2. Training Transfer Learning Model (MobileNetV2) <<<")
    print("="*50)
    transfer_model = build_transfer_model(num_classes=len(class_names)).to(device)
    optimizer_transfer = optim.Adam(transfer_model.classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_transfer = optim.lr_scheduler.StepLR(optimizer_transfer, step_size=5, gamma=0.5)

    transfer_model, transfer_history = train_model(
        transfer_model, train_loader, val_loader, criterion, optimizer_transfer,
        scheduler_transfer, num_epochs=12, device=device, model_name="MobileNetV2 (Transfer Learning)"
    )
    transfer_report, transfer_cm = evaluate_model(transfer_model, val_loader, class_names, device, "MobileNetV2 (Transfer Learning)")

    # Save transfer model weights
    transfer_save_path = os.path.join(OUTPUT_DIR, "transfer_mobilenet_lily.pth")
    torch.save(transfer_model.state_dict(), transfer_save_path)
    print(f"Saved MobileNetV2 weights to {transfer_save_path}")

    # ================= 3. Generate Comparative Visualizations =================
    plt.figure(figsize=(14, 5))

    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(scratch_history["train_loss"], label="Scratch Train Loss", linestyle="--", color="blue")
    plt.plot(scratch_history["val_loss"], label="Scratch Val Loss", color="blue")
    plt.plot(transfer_history["train_loss"], label="Transfer Train Loss", linestyle="--", color="green")
    plt.plot(transfer_history["val_loss"], label="Transfer Val Loss", color="green")
    plt.title("Training & Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(scratch_history["train_acc"], label="Scratch Train Acc", linestyle="--", color="blue")
    plt.plot(scratch_history["val_acc"], label="Scratch Val Acc", color="blue")
    plt.plot(transfer_history["train_acc"], label="Transfer Train Acc", linestyle="--", color="green")
    plt.plot(transfer_history["val_acc"], label="Transfer Val Acc", color="green")
    plt.title("Training & Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison curves to {plot_path}")

    # Save summary metrics to json
    results_summary = {
        "classes": class_names,
        "dataset_size": {
            "total": len(full_dataset),
            "train": train_size,
            "val": val_size
        },
        "scratch_cnn": {
            "best_val_accuracy": float(max(scratch_history["val_acc"])),
            "final_val_loss": float(scratch_history["val_loss"][-1]),
            "classification_report": scratch_report,
            "confusion_matrix": scratch_cm.tolist()
        },
        "transfer_learning": {
            "model_architecture": "MobileNetV2 (Pre-trained on ImageNet)",
            "best_val_accuracy": float(max(transfer_history["val_acc"])),
            "final_val_loss": float(transfer_history["val_loss"][-1]),
            "classification_report": transfer_report,
            "confusion_matrix": transfer_cm.tolist()
        }
    }

    summary_file = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to {summary_file}")

if __name__ == "__main__":
    run_experiment()
