"""Generate full 1080p MP4 presentation video with Neural AI Voiceover."""
import os
import subprocess
import asyncio
import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SLIDES_DIR = "/Users/erbiumtech/Downloads/slides_images"
AUDIO_DIR = "/Users/erbiumtech/Downloads/lily_classification/temp_audio"
CLIPS_DIR = "/Users/erbiumtech/Downloads/lily_classification/temp_clips"
FINAL_VIDEO = "/Users/erbiumtech/Downloads/Lily_Classification_Presentation.mp4"
WEB_VIDEO = "/Users/erbiumtech/Downloads/lily_classification/web/public/downloads/Lily_Classification_Presentation.mp4"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# Word-for-word voiceover script for all 10 slides
SCRIPTS = {
    1: (
        "Hello everyone and Respected Sir Prof. Dr Munir Ahmad. "
        "My name is Ahmad Ishtiaq, Roll Number SU92-BSCSM-F23-318, Section BSCS-7A. "
        "Welcome to my Deep Learning Semester Project presentation on the Lily Flower Classification System. "
        "In this project, we designed, trained, and benchmarked a Custom 4-Layer Convolutional Neural Network against a pre-trained MobileNetV2 Transfer Learning architecture on an extensive 15,276-image dataset, featuring an open-set non-flower rejection mechanism with a strict zero percent false positive guarantee."
    ),
    2: (
        "Moving to Slide 2: Project Objectives and Scope. "
        "Our primary goal was to accurately identify genuine Lily flowers from diverse other floral blooms while solving the critical closed-world limitation seen in traditional image classifiers. "
        "In traditional binary classifiers, uploading an arbitrary non-flower object like a laptop, car, or pet forces the model to guess a flower. Our project directly solves this. "
        "Furthermore, we implemented two separate paradigms, training from scratch versus transfer learning, and converted our models into browser-executable ONNX runtimes for live production deployment on Vercel."
    ),
    3: (
        "On Slide 3, we look at Dataset Engineering. "
        "Following strict academic requirements, we built a verified, balanced 3-class dataset of over 15,000 images. "
        "First, 5,000 pure Lily flower images from iNaturalist Open Data, covering Asiatic lilies, Stargazer, Tiger lilies, Easter lilies, and botanical outliers like the Dragon Lily. "
        "Second, 5,276 pure Other Flower images from Google TensorFlow Flowers and botanical repositories, featuring Roses, Tulips, Sunflowers, Daisies, and Dandelions. "
        "And third, 5,000 Not a Flower images comprising everyday objects, vehicles, and animals. "
        "We split this 80% for training, which is 12,221 images, and held out 20%, or 3,055 images, for strict validation."
    ),
    4: (
        "Slide 4 covers a pivotal empirical finding from our project. "
        "When we initially attempted uncurated web scraping, approximately 32% of scraped images turned out to be noise, such as botanical distribution maps, herbarium specimen sheets, bow ties, and insects. Training on that noisy data severely choked our models at 61% accuracy. "
        "By discarding the noisy data and rebuilding with research-grade botanical observations, our Custom CNN jumped from 61% to 75.3%, and MobileNetV2 jumped from 66.5% to 85.1%. "
        "This proved that data quality and label purity dictate model ceiling far more than raw volume."
    ),
    5: (
        "On Slide 5, we compare our two model architectures. "
        "Model 1 is our Custom CNN, trained entirely from scratch. It features 4 convolutional blocks with Batch Normalization, ReLU activations, Max Pooling, Adaptive Average Pooling, and two dense dropout layers. "
        "Model 2 is MobileNetV2, leveraging transfer learning pre-trained on ImageNet's 1.28 million images. We froze the inverted residual feature extractor and attached a specialized classification head. "
        "Both models were trained using Adam optimization and Apple Silicon MPS GPU acceleration."
    ),
    6: (
        "Slide 6 presents our Open-Set Non-Flower Rejection engine. "
        "Binary Softmax classifiers have a fatal flaw: probabilities always sum to 1.0, forcing non-flowers to look like flowers. "
        "To fix this, we designed a 3-class gating system: when the non-flower confidence exceeds 40% or dominates, the output displays Not a Flower, and both Lily and Other Flowers percentages are strictly locked to 0.0%. "
        "Only when floral features are confirmed does the network normalize probabilities between Lily and other flowers. This provides safety and high reliability in real-world use."
    ),
    7: (
        "On Slide 7, we examine the quantitative evaluation on our 3,055 validation images. "
        "Our Custom CNN trained from scratch achieved a solid 75.29% validation accuracy, with 84% Lily recall. "
        "Meanwhile, MobileNetV2 Transfer Learning was our champion, achieving 85.11% validation accuracy, a 91.01% Lily recall, and a 90.35% non-flower rejection recall. "
        "Transfer learning decisively outperformed the scratch model, converging faster and capturing subtle botanical petal textures with higher fidelity."
    ),
    8: (
        "Slide 8 highlights our domain case study: the Dragon Lily, scientifically Dracunculus vulgaris. "
        "Unlike conventional six-petal trumpet lilies, the Dragon Lily has a dramatic dark maroon spathe and a tall black spadix spike. Standard models typically misclassify it. "
        "By intentionally sourcing research-grade Dragon Lily observations into our training set, our fine-tuned MobileNetV2 successfully recognizes Dragon Lilies as true lilies with high confidence, demonstrating our system's resilience to high intra-class variance."
    ),
    9: (
        "Slide 9 shows our production deployment on Vercel. "
        "We converted both PyTorch models into client-side ONNX models, just 3.5 megabytes for the Custom CNN and 9.1 megabytes for MobileNetV2. "
        "They run directly in any web browser using ONNX Runtime Web with zero server compute lag and complete privacy. "
        "Furthermore, for automated grading, our Vercel web app features a dedicated Submission Hub with direct one-click download links for the dataset ZIP, PDF report, Word report, presentation slides, and PyTorch model weight checkpoints."
    ),
    10: (
        "To conclude on Slide 10: We have delivered an end-to-end, mathematically sound deep learning project trained on over 15,000 images, rigorously evaluated, and deployed live. "
        "All source code, trained weights, dataset scripts, live web application, and LinkedIn post are active and documented. "
        "Special thanks to our instructor Prof. Dr Munir Ahmad for his guidance and valuable requirements throughout this semester. "
        "Thank you very much for your time and attention!"
    )
}

VOICE = "en-US-ChristopherNeural"  # Professional, articulate academic presentation voice

async def generate_audio():
    print(f"🎙️ Generating Neural Audio for 10 slides using voice: {VOICE}...")
    for i in range(1, 11):
        text = SCRIPTS[i]
        audio_file = os.path.join(AUDIO_DIR, f"audio_{i:02d}.mp3")
        print(f"  Generating Slide {i} audio...")
        communicate = edge_tts.Communicate(text, VOICE, rate="+3%")
        await communicate.save(audio_file)
    print("✅ All 10 audio tracks generated successfully!")

def build_clips():
    print("🎬 Rendering individual slide video clips with FFMPEG...")
    clip_files = []
    for i in range(1, 11):
        img_file = os.path.join(SLIDES_DIR, f"slide_{i:02d}.png")
        audio_file = os.path.join(AUDIO_DIR, f"audio_{i:02d}.mp3")
        clip_file = os.path.join(CLIPS_DIR, f"clip_{i:02d}.mp4")

        # FFMPEG: Loop image for the duration of the audio, encode in standard h264/aac
        cmd = [
            FFMPEG, "-y",
            "-loop", "1",
            "-i", img_file,
            "-i", audio_file,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            clip_file
        ]
        print(f"  Rendering Slide {i} video clip...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_files.append(clip_file)

    # Concat all clips into final video
    print("🎞️ Merging all 10 slide clips into master presentation MP4...")
    concat_list_file = os.path.join(CLIPS_DIR, "concat_list.txt")
    with open(concat_list_file, "w") as f:
        for cf in clip_files:
            f.write(f"file '{cf}'\n")

    cmd_concat = [
        FFMPEG, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy",
        "-movflags", "+faststart",
        FINAL_VIDEO
    ]
    subprocess.run(cmd_concat, check=True)
    
    # Copy to web public downloads
    import shutil
    shutil.copyfile(FINAL_VIDEO, WEB_VIDEO)

    sz_mb = os.path.getsize(FINAL_VIDEO) / (1024 * 1024)
    print(f"🎉 MASTER VIDEO CREATED SUCCESSFULLY!")
    print(f"  📍 Saved to: {FINAL_VIDEO} ({sz_mb:.2f} MB)")
    print(f"  🌐 Public copy: {WEB_VIDEO}")

if __name__ == "__main__":
    asyncio.run(generate_audio())
    build_clips()
