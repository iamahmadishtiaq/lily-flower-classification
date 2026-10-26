import os
import io
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_defs import CustomLilyCNN, build_transfer_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES = ["Lily", "Other Flowers"]

device = torch.device("cpu")

# Load models once
scratch_model = CustomLilyCNN(num_classes=2)
scratch_path = os.path.join(MODELS_DIR, "custom_cnn_lily.pth")
if os.path.exists(scratch_path):
    scratch_model.load_state_dict(torch.load(scratch_path, map_location=device))
    scratch_model.eval()

transfer_model = build_transfer_model(num_classes=2)
transfer_path = os.path.join(MODELS_DIR, "transfer_mobilenet_lily.pth")
if os.path.exists(transfer_path):
    transfer_model.load_state_dict(torch.load(transfer_path, map_location=device))
    transfer_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lily Flower Classifier</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #f0f2f5; color: #1c1e21; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 30px 15px; }
    .card { background: #fff; width: 100%; max-width: 650px; border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
    h1 { font-size: 24px; font-weight: 700; color: #111827; margin-bottom: 8px; text-align: center; }
    p.subtitle { font-size: 14px; color: #6b7280; text-align: center; margin-bottom: 24px; }
    .dropzone { border: 2px dashed #d1d5db; border-radius: 12px; padding: 36px 20px; text-align: center; cursor: pointer; transition: all 0.2s; background: #fafafa; }
    .dropzone:hover { border-color: #10b981; background: #f0fdf4; }
    .dropzone input { display: none; }
    .dropzone-icon { font-size: 44px; margin-bottom: 12px; }
    .preview-container { margin-top: 20px; text-align: center; display: none; }
    .preview-img { max-height: 220px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .results { margin-top: 28px; display: none; }
    .model-box { background: #f9fafb; border-radius: 12px; padding: 18px; margin-bottom: 16px; border: 1px solid #e5e7eb; }
    .model-title { font-weight: 600; font-size: 15px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .badge { font-size: 12px; padding: 4px 10px; border-radius: 20px; font-weight: 600; }
    .badge-lily { background: #d1fae5; color: #065f46; }
    .badge-other { background: #fee2e2; color: #991b1b; }
    .progress-bar { background: #e5e7eb; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }
    .fill-lily { background: #10b981; }
    .fill-other { background: #ef4444; }
    .btn { display: block; width: 100%; padding: 12px; background: #10b981; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 18px; }
    .btn:hover { background: #059669; }
    .btn:disabled { background: #9ca3af; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🌸 Lily Flower Classifier</h1>
    <p class="subtitle">Scratch CNN vs. Transfer Learning (MobileNetV2)</p>

    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
      <div class="dropzone-icon">📷</div>
      <strong>Click to upload an image</strong> or drag and drop here
      <p style="font-size: 12px; color: #9ca3af; margin-top: 6px;">JPG, PNG, JPEG supported</p>
      <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
    </div>

    <div class="preview-container" id="previewContainer">
      <img id="previewImg" class="preview-img" alt="Preview">
      <button class="btn" id="predictBtn" onclick="runPrediction()">Run Prediction</button>
    </div>

    <div class="results" id="results">
      <!-- Custom CNN -->
      <div class="model-box">
        <div class="model-title">
          <span>🧠 Custom CNN (Trained from Scratch)</span>
          <span class="badge" id="scratchBadge">--</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 13px; color: #4b5563;">
          <span>Lily: <strong id="scratchLily">--%</strong></span>
          <span>Other Flowers: <strong id="scratchOther">--%</strong></span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" id="scratchFill" style="width: 0%;"></div>
        </div>
      </div>

      <!-- Transfer Learning -->
      <div class="model-box">
        <div class="model-title">
          <span>🚀 MobileNetV2 (Transfer Learning)</span>
          <span class="badge" id="transferBadge">--</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 13px; color: #4b5563;">
          <span>Lily: <strong id="transferLily">--%</strong></span>
          <span>Other Flowers: <strong id="transferOther">--%</strong></span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" id="transferFill" style="width: 0%;"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let base64Image = null;

    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(evt) {
        base64Image = evt.target.result;
        document.getElementById('previewImg').src = base64Image;
        document.getElementById('previewContainer').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').innerText = 'Run Prediction';
      };
      reader.readAsDataURL(file);
    }

    async function runPrediction() {
      const btn = document.getElementById('predictBtn');
      btn.disabled = true;
      btn.innerText = 'Analyzing image...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        // Update Scratch
        const sLily = data.scratch.probs.Lily;
        const sOther = data.scratch.probs["Other Flowers"];
        const sPred = data.scratch.prediction;
        document.getElementById('scratchBadge').innerText = sPred + ' (' + data.scratch.confidence + '%)';
        document.getElementById('scratchBadge').className = 'badge ' + (sPred === 'Lily' ? 'badge-lily' : 'badge-other');
        document.getElementById('scratchLily').innerText = sLily + '%';
        document.getElementById('scratchOther').innerText = sOther + '%';
        document.getElementById('scratchFill').style.width = sLily + '%';
        document.getElementById('scratchFill').className = 'progress-fill ' + (sPred === 'Lily' ? 'fill-lily' : 'fill-other');

        // Update Transfer
        const tLily = data.transfer.probs.Lily;
        const tOther = data.transfer.probs["Other Flowers"];
        const tPred = data.transfer.prediction;
        document.getElementById('transferBadge').innerText = tPred + ' (' + data.transfer.confidence + '%)';
        document.getElementById('transferBadge').className = 'badge ' + (tPred === 'Lily' ? 'badge-lily' : 'badge-other');
        document.getElementById('transferLily').innerText = tLily + '%';
        document.getElementById('transferOther').innerText = tOther + '%';
        document.getElementById('transferFill').style.width = tLily + '%';
        document.getElementById('transferFill').className = 'progress-fill ' + (tPred === 'Lily' ? 'fill-lily' : 'fill-other');

        document.getElementById('results').style.display = 'block';
      } catch (err) {
        alert('Prediction failed: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = 'Run Prediction';
      }
    }
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                b64_str = data["image"].split(",")[-1]
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                # Predict Scratch
                with torch.no_grad():
                    scratch_probs = F.softmax(scratch_model(tensor), dim=1)[0]
                    s_idx = torch.argmax(scratch_probs).item()

                # Predict Transfer
                with torch.no_grad():
                    transfer_probs = F.softmax(transfer_model(tensor), dim=1)[0]
                    t_idx = torch.argmax(transfer_probs).item()

                resp = {
                    "scratch": {
                        "prediction": CLASSES[s_idx],
                        "confidence": round(scratch_probs[s_idx].item() * 100, 1),
                        "probs": {
                            "Lily": round(scratch_probs[0].item() * 100, 1),
                            "Other Flowers": round(scratch_probs[1].item() * 100, 1)
                        }
                    },
                    "transfer": {
                        "prediction": CLASSES[t_idx],
                        "confidence": round(transfer_probs[t_idx].item() * 100, 1),
                        "probs": {
                            "Lily": round(transfer_probs[0].item() * 100, 1),
                            "Other Flowers": round(transfer_probs[1].item() * 100, 1)
                        }
                    }
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(("127.0.0.1", port), RequestHandler)
    print(f"Lily Classifier Web UI running at http://127.0.0.1:{port}")
    server.serve_forever()
