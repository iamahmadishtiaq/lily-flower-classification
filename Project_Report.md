# Lily Flower Classification — Semester Project Report

**Course / Subject:** Deep Learning  
**Project Type:** Semester Project  
**Student Name:** Ahmad Ishtiaq  
**Roll No:** SU92-BSCSM-F23-318  
**Section:** BSCS-7A  
**Instructor:** Prof. Dr Munir Ahmad  
**Date:** October 2026  

---

## 1. Project Overview

This project implements a **Binary Image Classifier** to distinguish **Lily flowers** from **Other Flowers** using two deep learning approaches:
1. A **Custom Convolutional Neural Network (CNN)** trained entirely from scratch.
2. A **Transfer Learning model** based on **MobileNetV2** pre-trained on ImageNet.

The objective is to compare both approaches and analyze their performance, limitations, and real-world applicability.

---

## 2. Dataset Collection (Self-Collected — No Kaggle Used)

### 2.1 Source & Methodology
As per assignment guidelines, **no Kaggle datasets were used**. A fully automated Python pipeline was written to:
- Query Wikipedia's open REST API across **66+ botanical article titles and search terms**.
- Download the TensorFlow Flowers dataset (official Google dataset — daisy, dandelion, roses, sunflowers, tulips).
- Filter, download, and verify images using PIL (Pillow).
- Reject corrupted, grayscale, or undersized (<80×80 px) images automatically.

### 2.2 Dataset Cleaning (Critical Step)
The initial dataset of ~1,093 images contained **354 non-flower images** that were accidentally downloaded from Wikipedia (botanical diagrams, distribution maps, habitat photos, portraits, icons, etc.). A **two-pass AI-powered cleaning pipeline** was implemented:

1. **Pass 1:** Used a pre-trained MobileNetV2 (ImageNet) to classify every image. Any image whose top-5 predictions did NOT include a flower-related ImageNet class was automatically deleted.
2. **Pass 2:** After downloading replacement images, the entire dataset was re-verified to ensure 100% flower-only content.

**Examples of removed non-flower images:**

| Removed Image | Model Prediction | Confidence |
|:-------------|:----------------|:-----------|
| `dark_lily_011.jpg` | bow tie | 92.1% |
| `dark_lily_033.jpg` | military uniform | 50.7% |
| `dark_lily_030.jpg` | beacon | 87.3% |
| `lily_0425.jpg` | pomegranate | 97.0% |
| `lily_0426.jpg` | lipstick | 99.1% |
| `other_flowers_0514.jpg` | neck brace | 74.9% |
| `other_flowers_0515.jpg` | proboscis monkey | 24.8% |
| `other_flowers_0489.jpg` | shower curtain | 70.5% |
| `other_flowers_0517.jpg` | monastery | 73.0% |
| `other_flowers_0438.jpg` | head cabbage | 98.7% |

> **Key Lesson:** Automated web scraping for datasets requires rigorous post-processing. Without AI-based verification, ~32% of our dataset contained irrelevant images that would have severely impacted model training.

### 2.3 Final Dataset Composition

| Category | Image Count | Species / Sources |
|:---------|:-----------:|:-----------------|
| **Lily (`dataset/lily`)** | **520** | Lilium candidum, Lilium lancifolium (Tiger Lily), Nymphaea (Water Lily), Zantedeschia (Calla Lily), Spathiphyllum (Peace Lily), Hemerocallis (Daylily), Dracunculus vulgaris (Dragon Lily), Gloriosa superba, Victoria amazonica, Hippeastrum, Convallaria (Lily of the Valley), + 55 more Lilium species |
| **Other Flowers (`dataset/other_flowers`)** | **520** | Rosa (Roses), Helianthus (Sunflower), Tulipa (Tulip), Phalaenopsis (Orchid), Tagetes (Marigold), Hibiscus, Nelumbo (Lotus), Dahlia, Chrysanthemum, Lavandula, Papaver (Poppy), Iris, Bellis (Daisy), Taraxacum (Dandelion), + TensorFlow Flowers dataset |
| **Total** | **1,040** | All AI-verified flower-only images |

### 2.4 Train / Validation Split
- **80% Training**: 832 images (with augmentation)
- **20% Validation**: 208 images
- **Split Method**: `random_split` with fixed seed (42) for reproducibility

### 2.5 Data Augmentation (Training Only)
To improve generalization and prevent overfitting:
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness ±0.2, contrast ±0.2)
- Resize to 224×224 pixels
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 3. Model Architectures

### 3.1 Model 1: Custom CNN (Trained from Scratch)

A 4-block Convolutional Neural Network designed and trained entirely from random initialization.

**Architecture Summary:**

```
Input (3×224×224)
→ Conv Block 1: Conv2d(3→32, 3×3) → BatchNorm → ReLU → MaxPool (→112×112)
→ Conv Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool (→56×56)
→ Conv Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool (→28×28)
→ Conv Block 4: Conv2d(128→256, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool (→4×4)
→ Flatten
→ Dropout(0.4) → Linear(4096→128) → ReLU → Dropout(0.3) → Linear(128→2)
Output: 2 classes (Lily / Other Flowers)
```

- **Optimizer**: Adam (lr=5×10⁻⁴, weight_decay=1×10⁻⁴)
- **Scheduler**: StepLR (step_size=7, gamma=0.5)
- **Epochs**: 15
- **Batch Size**: 16

### 3.2 Model 2: MobileNetV2 (Transfer Learning)

A pre-trained **MobileNetV2** (trained on ImageNet with 1.2M+ images) with its feature extractor **frozen** and a custom classification head fine-tuned on our lily dataset.

**Modifications:**
- All `model.features` layers: **frozen** (gradients disabled)
- New classifier head replaces original:
  ```
  Dropout(0.2) → Linear(1280→128) → ReLU → Dropout(0.2) → Linear(128→2)
  ```

- **Optimizer**: Adam (lr=1×10⁻³, applied only to classifier parameters)
- **Scheduler**: StepLR (step_size=5, gamma=0.5)
- **Epochs**: 12
- **Batch Size**: 16

---

## 4. Experiments & Results

### 4.1 Training Results

| Model | Dataset Size | Epochs | Best Val Accuracy | Final Val Loss |
|:------|:-----------:|:------:|:-----------------:|:--------------:|
| Custom CNN (Scratch) | 1,040 images | 15 | **61.06%** | 0.734 |
| MobileNetV2 (Transfer Learning) | 1,040 images | 12 | **83.65%** | 0.377 |

### 4.2 Evaluation Metrics

**Custom CNN (Scratch):**
```
               precision  recall  f1-score  support
         lily      0.59    0.51      0.55       96
other_flowers     0.62    0.70      0.66      112
     accuracy                        0.61      208
   macro avg      0.61    0.60      0.60      208
weighted avg      0.61    0.61      0.61      208
```

**Confusion Matrix:**
```
              Predicted
              Lily  Other
Actual Lily    49    47
Actual Other   34    78
```

**MobileNetV2 (Transfer Learning):**
```
               precision  recall  f1-score  support
         lily      0.78    0.91      0.84       96
other_flowers     0.91    0.78      0.84      112
     accuracy                        0.84      208
   macro avg      0.84    0.84      0.84      208
weighted avg      0.85    0.84      0.84      208
```

**Confusion Matrix:**
```
              Predicted
              Lily  Other
Actual Lily    87     9
Actual Other   25    87
```

### 4.3 Impact of Dataset Cleaning on Accuracy

| Model | Before Cleaning (dirty) | After Cleaning (clean) | Improvement |
|:------|:----------------------:|:---------------------:|:-----------:|
| Custom CNN | 59.6% | 61.06% | **+1.5%** |
| MobileNetV2 | 66.5% | **83.65%** | **+17.2%** |

> **Key Finding:** Cleaning the dataset (removing 354 non-flower images) improved MobileNetV2 accuracy by **17.2 percentage points**. This demonstrates that **data quality is more important than data quantity** in deep learning.

---

## 5. Problems Faced

### 5.1 Dataset Collection Challenges

| # | Problem | Cause | Solution Applied |
|:-:|:--------|:------|:----------------|
| 1 | **Non-flower images in dataset (~32% contamination)** | Wikipedia article pages contain all types of images — maps, diagrams, portraits, icons, habitat photos — not just flower photos | Built an AI-powered cleaning pipeline using pre-trained MobileNetV2 to verify every image is a flower. Removed 354 non-flower images. |
| 2 | Wikimedia API connection reset | Old SSL library incompatible with Wikimedia HTTPS | Switched to `requests` library with custom User-Agent headers |
| 3 | Low-quality / irrelevant images (icons, logos, distribution maps) | Wikipedia article images include UI elements and non-photographic content | Added extensive blacklist keyword filter (60+ terms) on file titles |
| 4 | Small initial dataset | Wikipedia category pages have limited unique images per species | Combined 66+ distinct botanical article titles + TensorFlow Flowers dataset to maximize image pool |
| 5 | Duplicate image URLs | Multiple Wikipedia articles link to same Wikimedia file | Used Python `set()` to deduplicate URLs before downloading |
| 6 | Images too small / corrupted | Some thumbnail URLs return tiny icons (<80px) or broken files | Added minimum size validation (80×80 px) and PIL verification |

### 5.2 Training Challenges

| # | Problem | Cause | Solution Applied |
|:-:|:--------|:------|:----------------|
| 1 | **Model accuracy was low on dirty dataset** | ~32% of images were not flowers (bow ties, monkeys, monasteries, etc.) — the model was trying to learn patterns from irrelevant images | Cleaned entire dataset using AI verification, retrained models. MobileNetV2 jumped from 66.5% to 83.65%. |
| 2 | Scratch CNN overfitting | Small dataset, model learning from scratch without prior knowledge | Added BatchNorm, Dropout layers (0.3-0.4), weight decay (1e-4), data augmentation (flip, rotation, color jitter) |
| 3 | Dragon Lily misclassified as "Other Flowers" | *Dracunculus vulgaris* has completely different visual structure (dark purple spathe + black spadix) vs. typical 6-petal lilies | Downloaded diverse Dragon Lily images and added to training dataset |
| 4 | Transfer model classifier architecture mismatch | `model_defs.py` had different layer sizes (256) than `train_models.py` (128), causing weight loading errors | Synchronized architecture definitions across all files |
| 5 | MPS (Apple Silicon) memory warnings | Processing large batches on M-series GPU | Reduced batch size to 16, used `torch.no_grad()` during validation |
| 6 | ONNX export failed | Missing `onnx` Python module | Installed `onnx` package separately |

### 5.3 Key Finding: Data Quality > Data Quantity

The most impactful discovery in this project was the effect of **dataset quality** on model performance:

- **Before cleaning:** 1,093 images (including 354 non-flower images) → MobileNetV2 could only reach **66.5% accuracy**
- **After cleaning:** 1,040 images (all verified flowers) → MobileNetV2 reached **83.65% accuracy**

Despite having **fewer total images** after cleaning, accuracy improved dramatically. This proves that **a smaller, cleaner dataset outperforms a larger, noisy one**.

### 5.4 Key Finding: Dragon Lily (Visual Outlier Problem)

The Dragon Lily (*Dracunculus vulgaris*) presented a unique challenge — confirmed by Google and Gemini to be a type of lily, yet visually it looks nothing like a typical lily:
- **Common Lily**: 6 open colorful petals in star/trumpet shape
- **Dragon Lily**: Dark purple spathe (funnel-shaped bract) + tall black spadix spike

After adding Dragon Lily images to the training set, the Transfer Learning model correctly classified it as **Lily with 90.51% confidence**, demonstrating the critical importance of **training data diversity**.

---

## 6. Comparison: Scratch CNN vs. Transfer Learning

| Aspect | Custom CNN (Scratch) | MobileNetV2 (Transfer Learning) |
|:-------|:--------------------:|:-------------------------------:|
| **Pre-trained weights** | ❌ None (random init) | ✅ ImageNet (1.2M images) |
| **Best Validation Accuracy** | 61.06% | **83.65%** |
| **Training time** | ~2 min 18 sec | ~1 min 18 sec |
| **Parameters trained** | All layers (~2.5M) | Only classifier head (~0.2M) |
| **Lily Recall** | 51.0% | **90.6%** |
| **Visual outlier handling** | ❌ Struggles | ✅ Better generalization |
| **Small dataset performance** | Moderate | Higher |
| **Production suitability** | Research/educational | ✅ Deployment-ready |

**Conclusion:** Transfer Learning (MobileNetV2) significantly outperforms the custom CNN, achieving **22.6 percentage points** higher accuracy. The pre-trained features from ImageNet provide a strong foundation for flower classification even with a relatively small dataset.

---

## 7. Web Application (Vercel Deployment)

The trained models were converted to **ONNX format** for client-side browser inference:

| Model | ONNX File | Size |
|:------|:----------|:-----|
| Custom CNN | `custom_cnn_lily.onnx` | 3.5 MB |
| MobileNetV2 | `mobilenetv2_lily.onnx` | 9.1 MB |

The web app uses **ONNX Runtime Web** to run inference entirely in the browser — no server required. Users can drag-and-drop or upload a flower image and get instant classification results.

**Deployment:** Static site hosted on **Vercel** (Root Directory: `web`, Framework Preset: Other).

---

## 8. Project Structure

```
lily_classification/
├── dataset/
│   ├── lily/                 ← 520 verified lily flower images
│   └── other_flowers/        ← 520 verified other flower images
├── models/
│   ├── custom_cnn_lily.pth
│   ├── transfer_mobilenet_lily.pth
│   ├── training_comparison.png
│   └── experiment_results.json
├── web/
│   ├── public/
│   │   ├── index.html        ← Web UI (drag & drop classifier)
│   │   └── models/           ← ONNX models for browser inference
│   ├── package.json
│   └── vercel.json
├── Lily_Flower_Classification.ipynb  ← Google Colab notebook
├── Project_Report.md                 ← This report
├── LinkedIn_Post_and_Submission.md
├── train_models.py
├── model_defs.py
├── predict.py
├── download_dataset.py
├── clean_and_redownload.py   ← AI-powered dataset cleaner
├── topup_dataset.py          ← Dataset top-up to 500+ per class
└── README.md
```

---

## 9. Technologies Used

| Tool / Library | Purpose |
|:--------------|:--------|
| **Python 3.9** | Core language |
| **PyTorch 2.x** | Deep learning framework |
| **TorchVision** | Pre-trained models, transforms, datasets |
| **ONNX** | Model export for web deployment |
| **ONNX Runtime Web** | Browser-based model inference |
| **Pillow (PIL)** | Image loading and preprocessing |
| **Requests** | Wikipedia API image downloading |
| **Matplotlib** | Training curve visualization |
| **Scikit-learn** | Classification report, confusion matrix |
| **Google Colab** | Cloud training environment (GPU) |
| **Apple MPS** | Local GPU acceleration (M-series chip) |
| **Vercel** | Web application hosting |
| **GitHub** | Version control and code hosting |

---

## 10. Conclusion

This project successfully demonstrated:
1. Building a **self-collected, diverse flower dataset** (1,040 verified images) without Kaggle.
2. Discovering and solving a **major dataset quality problem** — 32% of initially scraped images were not flowers, and removing them improved accuracy by up to 17%.
3. Designing and training a **Custom CNN from scratch** achieving 61% validation accuracy.
4. Applying **Transfer Learning (MobileNetV2)** which achieved **83.65% accuracy** — outperforming the scratch model by 22.6 percentage points.
5. Encountering and solving a **real-world visual outlier problem** (Dragon Lily) — demonstrating the importance of training data diversity.
6. Deploying the model as a **web application using ONNX** for zero-latency client-side inference on Vercel.

**Key Takeaway:** Data quality matters more than quantity. A clean dataset of 1,040 images outperformed a noisy dataset of 1,093 images by a significant margin. Transfer learning combined with clean data provides the best results for small-to-medium datasets.

---

*Report prepared by Ahmad Ishtiaq (Roll No: SU92-BSCSM-F23-318, Section: BSCS-7A) as part of the Deep Learning Semester Project under the supervision of Prof. Dr Munir Ahmad.*
