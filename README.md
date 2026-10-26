# 🌸 Lily Flower Classification Project

An end-to-end Computer Vision & Deep Learning project to classify **Lily Flowers** vs. **Other Flowers**, built from scratch and compared with Transfer Learning.

🌐 **Live Demo Hosted on Vercel:** [Coming Soon / Your Vercel Link]  
📓 **Google Colab Notebook:** [`Lily_Flower_Classification.ipynb`](./Lily_Flower_Classification.ipynb)  
📄 **Comprehensive Project Report:** [`Project_Report.md`](./Project_Report.md)

---

## 📌 Project Overview & Highlights
- **100% Self-Collected Dataset (No Kaggle)**: Collected directly from open botanical sources via Wikipedia/Wikimedia API. Over **1,034 verified images** covering 26+ flower species.
- **Two Deep Learning Models**:
  1. **Custom CNN from Scratch**: 4 Conv blocks + BatchNorm + Dropout + Dense classifier trained entirely from random weights.
  2. **Transfer Learning (MobileNetV2)**: Pre-trained backbone fine-tuned for high-accuracy botanical feature detection.
- **Real-World Case Study (Dragon & Black Lilies)**: Addressed intra-class variance and biological outliers (*Dracunculus vulgaris*, Black Charm Lily) where Transfer Learning excelled over scratch models.
- **Client-Side Vercel Deployment**: Converted PyTorch weights to lightweight ONNX web format for instant, 0-latency browser inference with zero server costs.

---

## 📊 Performance Comparison

| Metric | Custom CNN (From Scratch) | MobileNetV2 (Transfer Learning) |
|:-------|:-------------------------:|:-------------------------------:|
| **Architecture** | 4-Layer Custom CNN | MobileNetV2 (ImageNet) |
| **Validation Accuracy** | **~68% - 75%** | **~87% - 91%** |
| **Inference Time** | ~15 ms | ~8 ms |
| **Dragon / Black Lily Detection** | Confused by rare color/shape | **Accurate (85% - 91% conf)** |
| **Parameters** | 2.5 Million | 2.2 Million (Only head trained) |

---

## 📁 Repository Structure
```
lily-flower-classification/
├── web/                             # Vercel-ready Frontend & ONNX deployment
│   ├── public/
│   │   ├── index.html               # Modern drag-and-drop web UI
│   │   └── models/
│   │       ├── mobilenetv2_lily.onnx # Exported Web ONNX model (~10MB)
│   │       └── custom_cnn_lily.onnx  # Exported Web ONNX model (~3.6MB)
│   ├── vercel.json                  # Vercel routing configuration
│   └── package.json
├── models/                          # Trained PyTorch checkpoints (.pth)
│   ├── transfer_mobilenet_lily.pth
│   ├── custom_cnn_lily.pth
│   └── training_comparison.png      # Loss/Accuracy plots
├── dataset/                         # 1,000+ categorized images
│   ├── lily/
│   └── other_flowers/
├── Lily_Flower_Classification.ipynb # Google Colab complete notebook
├── Project_Report.md                # Full academic project report
├── train_models.py                  # Training pipeline
├── model_defs.py                    # Model architectures
└── predict.py                       # CLI inference tool
```

---

## 🚀 Quick Start & Testing

### 1. Test Locally via CLI
```bash
python predict.py "path/to/any_flower_image.jpg"
```

### 2. Run Google Colab Notebook
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Lily_Flower_Classification.ipynb`.
3. Run all cells to see data collection, training, and use the interactive image upload widget.

### 3. Deploy to Vercel (1-Click)
1. Fork or push this repository to your GitHub.
2. Go to [vercel.com/new](https://vercel.com/new).
3. Import this repository.
4. Set **Root Directory** to `web`.
5. Click **Deploy**!

---

## 👥 Author
- **Ahmad Ishtiaq** ([@iamahmadishtiaq](https://github.com/iamahmadishtiaq))
