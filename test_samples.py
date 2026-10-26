import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_defs import CustomLilyCNN, build_transfer_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CLASSES = ["lily", "other_flowers"]

def run_sample_tests(num_samples_per_class=3):
    device = torch.device("cpu")

    # Load models
    scratch_model = CustomLilyCNN(num_classes=2)
    scratch_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "custom_cnn_lily.pth"), map_location=device))
    scratch_model.eval()

    transfer_model = build_transfer_model(num_classes=2)
    transfer_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "transfer_mobilenet_lily.pth"), map_location=device))
    transfer_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n" + "="*70)
    print("      SAMPLE TESTING: MODEL RESULTS ON RANDOM FLOWER IMAGES")
    print("="*70)

    for true_class in CLASSES:
        folder = os.path.join(DATASET_DIR, true_class)
        all_imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        chosen = random.sample(all_imgs, min(num_samples_per_class, len(all_imgs)))

        print(f"\n>>> ACTUAL CATEGORY: [{true_class.upper()}] (Testing {len(chosen)} images)")
        print("-" * 70)

        for img_name in chosen:
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                s_probs = F.softmax(scratch_model(tensor), dim=1)[0]
                s_pred = CLASSES[torch.argmax(s_probs).item()]
                s_conf = s_probs[torch.argmax(s_probs).item()].item() * 100

                t_probs = F.softmax(transfer_model(tensor), dim=1)[0]
                t_pred = CLASSES[torch.argmax(t_probs).item()]
                t_conf = t_probs[torch.argmax(t_probs).item()].item() * 100

            s_icon = "✅" if s_pred == true_class else "❌"
            t_icon = "✅" if t_pred == true_class else "❌"

            print(f"File: {img_name:<20}")
            print(f"   Scratch CNN     -> {s_icon} Predicted: {s_pred:<14} (Confidence: {s_conf:.1f}%)")
            print(f"   Transfer Model  -> {t_icon} Predicted: {t_pred:<14} (Confidence: {t_conf:.1f}%)")
            print()

    print("="*70 + "\n")

if __name__ == "__main__":
    run_sample_tests(num_samples_per_class=3)
