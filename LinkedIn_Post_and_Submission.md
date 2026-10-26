# 📢 LinkedIn Post & Submission Materials

## 1. LinkedIn Post Draft
*Copy & paste this directly to your LinkedIn. Tag your instructor where indicated.*

```
🌸 Excited to share my latest Deep Learning & Computer Vision project: Lily Flower Classifier! 🌿

As part of our coursework, I developed an end-to-end binary classification system capable of distinguishing Lily flowers from other flower species, tackling real-world computer vision challenges along the way.

🔑 Key Project Highlights:

✅ 100% Self-Collected Dataset: In strict compliance with guidelines (no Kaggle), collected 1,040 flower images across 66+ botanical species via Wikipedia API & TensorFlow Flowers dataset.

✅ AI-Powered Data Cleaning: Discovered that ~32% of web-scraped images were NOT flowers (maps, diagrams, even monkey photos!). Built an automated cleaning pipeline using pre-trained MobileNetV2 to verify every image — removing 354 non-flower images. This single step improved model accuracy by 17%!

✅ Comparative Architecture Study:
  1. Custom 4-Layer CNN built and trained from scratch — 61% val accuracy
  2. MobileNetV2 Transfer Learning fine-tuned on custom data — 84% val accuracy 🚀

✅ Key Insight: Data Quality > Data Quantity — A smaller, cleaner dataset (1,040 images) dramatically outperformed the original noisy dataset (1,093 images). Cleaning the data improved MobileNetV2 from 66% to 84%!

✅ Visual Outlier Discovery: Tackled rare species like Dragon Lily (Dracunculus vulgaris) which looks nothing like a typical lily — demonstrating why training data diversity matters.

✅ Production Deployment: Converted PyTorch weights to ONNX format for client-side, zero-latency inference hosted live on Vercel!

🔗 Live Web App: [Your Vercel URL Here]
💻 GitHub: https://github.com/iamahmadishtiaq/lily-flower-classification
📓 Colab Notebook: Included in repository with interactive image tester!

Special thanks to my instructor [@Instructor Name] for the guidance throughout this assignment.

#MachineLearning #DeepLearning #ComputerVision #PyTorch #TransferLearning #AI #WebML #Vercel #ONNX #DataQuality #StudentProject
```

---

## 2. Google Form Submission Checklist
When filling out the Google Form submission, use these details:

| Field | Content to Paste |
|:------|:----------------|
| **Project Title** | Lily Flower Classification (Scratch CNN vs. Transfer Learning) |
| **GitHub Repository URL** | `https://github.com/iamahmadishtiaq/lily-flower-classification` |
| **Live Vercel URL** | `https://lily-flower-classification.vercel.app` (or your custom Vercel link) |
| **Colab Notebook** | Included in GitHub repo root (`Lily_Flower_Classification.ipynb`) |
| **Project Report** | `Project_Report.md` (Included in repo) |
| **Short Description** | An end-to-end Computer Vision project featuring a 1,040 self-collected & AI-verified botanical dataset (no Kaggle), an automated data cleaning pipeline that removed 354 non-flower images improving accuracy by 17%, comparative evaluation between a scratch CNN (61%) and MobileNetV2 Transfer Learning (84%), and an ONNX-powered web app deployed to Vercel. |
