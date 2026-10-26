import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_defs import CustomLilyCNN, build_transfer_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES = ["Lily", "Other Flowers"]

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found: '{image_path}'")
        return

    device = torch.device("cpu") # CPU is instant for single image inference

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
        scratch_model = CustomLilyCNN(num_classes=2)
        scratch_model.load_state_dict(torch.load(scratch_path, map_location=device))
        scratch_model.eval()
        with torch.no_grad():
            probs = F.softmax(scratch_model(tensor), dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_name = CLASSES[pred_idx]
            conf = probs[pred_idx].item() * 100
            print(f"[Model 1: Custom CNN (Scratch)]")
            print(f"  Prediction : {pred_name}")
            print(f"  Confidence : {conf:.2f}% (Lily: {probs[0]*100:.1f}% | Other: {probs[1]*100:.1f}%)")
    print("-" * 55)

    # 2. Transfer Learning Model
    transfer_path = os.path.join(MODELS_DIR, "transfer_mobilenet_lily.pth")
    if os.path.exists(transfer_path):
        transfer_model = build_transfer_model(num_classes=2)
        transfer_model.load_state_dict(torch.load(transfer_path, map_location=device))
        transfer_model.eval()
        with torch.no_grad():
            probs = F.softmax(transfer_model(tensor), dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_name = CLASSES[pred_idx]
            conf = probs[pred_idx].item() * 100
            print(f"[Model 2: MobileNetV2 (Transfer Learning)]")
            print(f"  Prediction : {pred_name}")
            print(f"  Confidence : {conf:.2f}% (Lily: {probs[0]*100:.1f}% | Other: {probs[1]*100:.1f}%)")
    print("="*55 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        # Default test if no arg passed
        sample = os.path.join(BASE_DIR, "dataset", "lily", "lily_0001.jpg")
        print("No image path provided, testing on sample lily image:")
        predict(sample)
