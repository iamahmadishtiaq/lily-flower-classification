# 🎙️ Video Presentation Voiceover Script (Slide-by-Slide)

**Project:** 🌸 Lily Flower Classification System with Open-Set Non-Flower Rejection  
**Course:** Deep Learning Semester Project  
**Student Name:** Ahmad Ishtiaq  
**Roll No:** SU92-BSCSM-F23-318  
**Section:** BSCS-7A  
**Instructor / Supervisor:** Prof. Dr Munir Ahmad  
**Estimated Recording Time:** 3 – 5 minutes  

---

## 💡 Quick Recording Instructions for Ahmad:
1. Open `Lily_Classification_Presentation.pptx` in Microsoft PowerPoint, Google Slides, or Keynote.
2. In PowerPoint, click **Record > From Beginning** (or use **Loom**, **OBS Studio**, or **QuickTime Player Screen Recording**).
3. Speak clearly into your microphone as you move from Slide 1 to Slide 10, reading the text below for each slide.
4. Export or save the video file (`.mp4`) and upload it to YouTube as **Public** or **Unlisted**.
5. Copy your YouTube video URL and add it to your submission form and website!

---

### 🎬 Slide 1: Title & Introduction (Duration: ~25s)
> *"Hello everyone and Respected Sir Prof. Dr Munir Ahmad.*  
> *My name is Ahmad Ishtiaq, Roll Number SU92-BSCSM-F23-318, from Section BSCS-7A.*  
> *Welcome to my Deep Learning Semester Project presentation on the **Lily Flower Classification System**.*  
> *In this project, we designed, trained, and benchmarked a **Custom 4-Layer Convolutional Neural Network** against a pre-trained **MobileNetV2 Transfer Learning architecture** on an extensive 15,276-image dataset, featuring an innovative **Open-Set Non-Flower Rejection mechanism** with a strict 0.0% false-positive guarantee."*

---

### 🎬 Slide 2: Project Objectives & Scope (Duration: ~30s)
> *"Moving to Slide 2: Project Objectives.*  
> *Our primary goal was to accurately identify genuine Lily flowers (*Lilium* species) from diverse other floral blooms while solving the critical closed-world limitation seen in traditional image classifiers.*  
> *In traditional classifiers, uploading an arbitrary non-flower object like a laptop, car, or pet forces the model to guess a flower. Our project directly solves this.*  
> *Furthermore, we implemented two separate paradigms—training from scratch versus transfer learning—and converted our models into browser-executable ONNX runtimes for live production deployment on Vercel."*

---

### 🎬 Slide 3: Dataset Engineering (15,276 Images) (Duration: ~35s)
> *"On Slide 3, we look at Dataset Engineering.*  
> *Following strict academic requirements, we built a verified, balanced 3-class dataset of over 15,000 images:*  
> *First, **5,000 pure Lily flower images** from iNaturalist Open Data, covering Asiatic lilies, Stargazer, Tiger lilies, Easter lilies, and botanical outliers like the Dragon Lily.*  
> *Second, **5,276 pure Other Flower images** from Google TensorFlow Flowers and botanical repositories, featuring Roses, Tulips, Sunflowers, Daisies, and Dandelions.*  
> *And third, **5,000 Not-a-Flower images** comprising everyday objects, vehicles, and animals.*  
> *We split this 80% for training—which is 12,221 images—and held out 20%, or 3,055 images, for strict validation."*

---

### 🎬 Slide 4: Data Quality Challenge & Scraper Contamination (Duration: ~30s)
> *"Slide 4 covers a pivotal empirical finding from our project.*  
> *When we initially attempted uncurated web scraping, approximately 32% of scraped images turned out to be noise—botanical distribution maps, herbarium specimen sheets, bow ties, and insects. Training on that noisy data severely choked our models at 61% accuracy.*  
> *By discarding the noisy data and rebuilding with research-grade botanical observations, our Custom CNN jumped from 61% to 75.3%, and MobileNetV2 jumped from 66.5% to 85.1%.*  
> *This proved that data quality and label purity dictate model ceiling far more than raw volume."*

---

### 🎬 Slide 5: Deep Learning Architectures Comparison (Duration: ~35s)
> *"On Slide 5, we compare our two model architectures:*  
> *Model 1 is our **Custom CNN**, trained entirely from scratch. It features 4 convolutional blocks with Batch Normalization, ReLU activations, Max Pooling, Adaptive Average Pooling, and two dense dropout layers.*  
> *Model 2 is **MobileNetV2**, leveraging transfer learning pre-trained on ImageNet’s 1.28 million images. We froze the inverted residual feature extractor and attached a specialized classification head.*  
> *Both models were trained using Adam optimization and Apple Silicon MPS GPU acceleration."*

---

### 🎬 Slide 6: Open-Set Non-Flower Rejection (0% Guarantee) (Duration: ~35s)
> *"Slide 6 presents our Open-Set Non-Flower Rejection engine.*  
> *Binary Softmax classifiers have a fatal flaw: probabilities always sum to 1.0, forcing non-flowers to look like flowers.*  
> *To fix this, we designed a 3-class gating system: when the non-flower confidence exceeds 40% or dominates, the output displays 'Not a Flower', and both Lily and Other Flowers percentages are strictly locked to **0.0%**.*  
> *Only when floral features are confirmed does the network normalize probabilities between Lily and other flowers. This provides safety and high reliability in real-world use."*

---

### 🎬 Slide 7: Experimental Results & Metrics (Duration: ~35s)
> *"On Slide 7, we examine the quantitative evaluation on our 3,055 validation images:*  
> *Our Custom CNN trained from scratch achieved a solid **75.29% validation accuracy**, with 84% Lily recall.*  
> *Meanwhile, MobileNetV2 Transfer Learning was our champion, achieving **85.11% validation accuracy**, a **91.01% Lily recall**, and a **90.35% non-flower rejection recall**.*  
> *Transfer learning decisively outperformed the scratch model, converging faster and capturing subtle botanical petal textures with higher fidelity."*

---

### 🎬 Slide 8: Botanical Outlier Study — The Dragon Lily (Duration: ~30s)
> *"Slide 8 highlights our domain case study: the **Dragon Lily** (*Dracunculus vulgaris*).*  
> *Unlike conventional six-petal trumpet lilies, the Dragon Lily has a dramatic dark maroon spathe and a tall black spadix spike. Standard models typically misclassify it.*  
> *By intentionally sourcing research-grade Dragon Lily observations into our training set, our fine-tuned MobileNetV2 successfully recognizes Dragon Lilies as true lilies with high confidence, demonstrating our system's resilience to high intra-class variance."*

---

### 🎬 Slide 9: Production Deployment & Crawler Compatibility (Duration: ~30s)
> *"Slide 9 shows our production deployment on Vercel.*  
> *We converted both PyTorch models into client-side ONNX models—just 3.5 MB for the Custom CNN and 9.1 MB for MobileNetV2.*  
> *They run directly in any web browser using ONNX Runtime Web with zero server compute lag and complete privacy.*  
> *Furthermore, for automated grading, our Vercel web app features a dedicated Submission Hub with direct one-click download links for the dataset ZIP, PDF report, Word report, presentation slides, and PyTorch model weight checkpoints (.pth)."*

---

### 🎬 Slide 10: Conclusion & Summary (Duration: ~25s)
> *"To conclude on Slide 10:*  
> *We have delivered an end-to-end, mathematically sound deep learning project trained on over 15,000 images, rigorously evaluated, and deployed live.*  
> *All source code, trained weights, dataset scripts, live web application, and LinkedIn post are active and documented.*  
> *Special thanks to our instructor Prof. Dr Munir Ahmad for his guidance and valuable requirements throughout this semester.*  
> *Thank you very much for your time and attention!"*

---
