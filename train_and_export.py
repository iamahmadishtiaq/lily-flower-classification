"""
Full pipeline: Train models on clean dataset, export ONNX, update web app.
"""
import os
import sys

# Run training
print("="*60)
print("🏋️ STEP 1: Training both models on clean dataset...")
print("="*60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Import and run training
from train_models import run_experiment
if __name__ == '__main__':
    run_experiment()

# Export to ONNX
print("\n" + "="*60)
print("📦 STEP 2: Exporting models to ONNX for web app...")
print("="*60)

import torch
import torch.nn as nn
from torchvision import models

MODELS_DIR = os.path.join(BASE_DIR, "models")
WEB_MODELS_DIR = os.path.join(BASE_DIR, "web", "public", "models")
os.makedirs(WEB_MODELS_DIR, exist_ok=True)

# ─── Export Custom CNN ───
print("\n📦 Exporting Custom CNN to ONNX...")

class CustomLilyCNN(nn.Module):
    def __init__(self, num_classes=3):
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

cnn_model = CustomLilyCNN(num_classes=3)
cnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "custom_cnn_lily.pth"), map_location="cpu"))
cnn_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
cnn_onnx_path = os.path.join(WEB_MODELS_DIR, "custom_cnn_lily.onnx")
torch.onnx.export(
    cnn_model, dummy_input, cnn_onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
print(f"  ✅ Custom CNN ONNX saved: {cnn_onnx_path} ({os.path.getsize(cnn_onnx_path) / (1024*1024):.1f} MB)")

# ─── Export MobileNetV2 ───
print("\n📦 Exporting MobileNetV2 to ONNX...")

def build_transfer_model(num_classes=3):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model

transfer_model = build_transfer_model(num_classes=3)
transfer_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "transfer_mobilenet_lily.pth"), map_location="cpu"))
transfer_model.eval()

mobilenet_onnx_path = os.path.join(WEB_MODELS_DIR, "mobilenetv2_lily.onnx")
torch.onnx.export(
    transfer_model, dummy_input, mobilenet_onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
print(f"  ✅ MobileNetV2 ONNX saved: {mobilenet_onnx_path} ({os.path.getsize(mobilenet_onnx_path) / (1024*1024):.1f} MB)")

print("\n" + "="*60)
print("✅ ALL DONE! Models trained and exported successfully!")
print("="*60)
