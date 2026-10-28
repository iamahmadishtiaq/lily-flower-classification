# 📢 LinkedIn Post & Submission Materials

## 1. LinkedIn Post Caption (Copy & Paste)

```markdown
🌸 Excited to unveil Florix AI — an end-to-end Deep Learning & Botanical Vision system capable of classifying Lily flowers with real-world Open-Set Non-Flower Rejection! 🌿

As part of my Deep Learning Semester Project under the supervision of Prof. Dr Munir Ahmad, I tackled one of the most prominent flaws in production computer vision: the Closed-World Softmax Dilemma.

Traditional binary vision models force probabilities to sum to 1.0. When an arbitrary non-flower object (such as a sports car, laptop, or animal) is input, standard models falsely label it as a flower with high confidence. Florix AI solves this with a strict 3-class open-set gating engine: whenever a non-flower entity is detected, both Lily and Other Flowers outputs strictly collapse to 0.0%! 🛡️

Here is what was engineered in this project:

📦 1. 15,276 Verified Botanical Images (Zero Scraper Noise):
Following strict academic instructions, no pre-made Kaggle sets were used. The dataset was collected and verified across 3 balanced distributions:
• 5,000 Verified Lilies (Lilium Genus: Asiatic, Stargazer, Tiger, Easter, Madonna) + Dragon Lily (Dracunculus vulgaris) from iNaturalist Open Data (AWS S3).
• 5,276 Other Flower Blooms (Roses, Tulips, Sunflowers, Daisies, Dandelions) from TF Flowers.
• 5,000 Non-Flower Objects (Mammals, vehicles, furniture, electronics) for open-set calibration.

💡 2. The Data Quality Discovery (+18.6% Accuracy Leap):
During initial web scraping, ~32% of scraped data consisted of noise (cladograms, pressed herbarium leaves, bow ties, medals). Removing this label noise and rebuilding with research-grade botanical flora caused MobileNetV2 validation accuracy to surge from 66.5% → 85.11%! Proof that Data Quality is far more critical than raw volume.

🧠 3. Comparative Architecture Study (N=3,055 Held-Out Validation):
• Model 1: Custom 4-Layer CNN (Trained from Scratch) — 75.29% Val Accuracy (83.96% Lily recall, ~580K params).
• Model 2: MobileNetV2 Transfer Learning (Pre-trained on ImageNet) — 85.11% Val Accuracy 🏆 (91.01% Lily recall, 90.35% non-flower rejection recall, 91.35% precision).

🌿 4. The Dragon Lily Challenge (High Intra-Class Variance):
*Dracunculus vulgaris* violates standard 6-petal trumpet lily morphology with its dark spathe and tall spadix spike. By curating verified observations directly into training, MobileNetV2 identifies Dragon Lilies as authentic lilies with >90% confidence!

⚡ 5. Production In-Browser Deployment on Vercel:
Exported trained PyTorch checkpoints to ONNX Web format (3.5 MB & 9.1 MB), executing client-side directly in the browser via ONNX Runtime Web (WASM). Zero backend server lag, zero compute costs, and 100% user data privacy!

🔗 Live Artifacts & Project Links:
🌐 Live Web App (Vercel): [Paste your Vercel Link here]
💻 GitHub Repository: https://github.com/iamahmadishtiaq/lily-flower-classification
🎬 YouTube Presentation Video: https://youtu.be/37qunhL2mKY

All project artifacts (Dataset ZIP, PDF/Word Research Reports, PPTX Slides, and PyTorch .pth model weights) are available directly on the web application.

Special thanks to our supervisor Prof. Dr Munir Ahmad for his mentorship and guidance throughout this semester project!

👨‍🎓 Student: Ahmad Ishtiaq
🔢 Roll No: SU92-BSCSM-F23-318
📚 Section: BSCS-7A (Department of Computer Science)

#DeepLearning #ComputerVision #MachineLearning #PyTorch #TransferLearning #MobileNetV2 #AI #WebML #ONNX #Vercel #FlowerClassification #DataQuality #DataScience #ArtificialIntelligence #SemesterProject
```
