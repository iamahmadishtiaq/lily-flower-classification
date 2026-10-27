"""
Inference script for Lily Flower Classifier with Open-Set Non-Flower Rejection:
- If input is NOT a flower (car, animal, human, object) -> Lily: 0.0%, Other Flowers: 0.0%
- If input IS a flower -> Accurate Lily % vs Other Flowers %
"""
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_defs import CustomLilyCNN, build_transfer_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES = ["lily", "not_a_flower", "other_flowers"]

def evaluate_probs(probs, model_title="Model"):
    p_lily = probs[0].item()
    p_not_flower = probs[1].item()
    p_other = probs[2].item()

    print(f"[{model_title}]")
    # Non-flower detection threshold
    if p_not_flower > 0.40 or (p_not_flower > p_lily and p_not_flower > p_other):
        print("  Status     : Non-Flower Object Detected 🚫")
        print("  Prediction : Not a Flower")
        print(f"  Confidence : {p_not_flower * 100:.1f}% Non-Flower")
        print("  Lily %     : 0.0%")
        print("  Other %    : 0.0%")
    else:
        # Normalize between Lily and Other Flowers
        flower_sum = p_lily + p_other
        lily_pct = (p_lily / flower_sum) * 100.0 if flower_sum > 0 else 50.0
        other_pct = (p_other / flower_sum) * 100.0 if flower_sum > 0 else 50.0

        if lily_pct >= other_pct:
            pred_name = "Lily 🌸"
            conf = lily_pct
        else:
            pred_name = "Other Flowers 🌼"
            conf = other_pct

        print("  Status     : Flower Detected ✅")
        print(f"  Prediction : {pred_name}")
        print(f"  Confidence : {conf:.2f}%")
        print(f"  Lily %     : {lily_pct:.1f}%")
        print(f"  Other %    : {other_pct:.1f}%")

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found: '{image_path}'")
        return

    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    tensor = transform(img).unsqueeze(0).to(device)

    print("\n" + "="*55)
    print(f"  Testing Image: {os.path.basename(image_path)}")
    print("="*55)

    # 1. Custom CNN
    scratch_path = os.path.join(MODELS_DIR, "custom_cnn_lily.pth")
    if os.path.exists(scratch_path):
        scratch_model = CustomLilyCNN(num_classes=3)
        scratch_model.load_state_dict(torch.load(scratch_path, map_location=device))
        scratch_model.eval()
        with torch.no_grad():
            probs = F.softmax(scratch_model(tensor), dim=1)[0]
            evaluate_probs(probs, "Model 1: Custom CNN (Scratch)")
    print("-" * 55)

    # 2. Transfer Learning Model
    transfer_path = os.path.join(MODELS_DIR, "transfer_mobilenet_lily.pth")
    if os.path.exists(transfer_path):
        transfer_model = build_transfer_model(num_classes=3)
        transfer_model.load_state_dict(torch.load(transfer_path, map_location=device))
        transfer_model.eval()
        with torch.no_grad():
            probs = F.softmax(transfer_model(tensor), dim=1)[0]
            evaluate_probs(probs, "Model 2: MobileNetV2 (Transfer Learning)")
    print("="*55 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        sample = os.path.join(BASE_DIR, "dataset", "lily", "lily_0001.jpg")
        print("Testing on sample lily image:")
        predict(sample)
