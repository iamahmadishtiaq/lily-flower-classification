"""
Clean dataset: remove non-flower images and re-download clean flower-only images.
Uses pretrained MobileNetV2 (ImageNet) to verify each image is a flower.
"""

import os
import sys
import shutil
import time
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ─── ImageNet flower-related class indices ───
# These are all the flower/plant classes in ImageNet-1K
FLOWER_CLASSES = {
    985: "daisy",
    986: "yellow lady's slipper",
    987: "corn",  # not flower, skip
    988: "acorn",  # not flower, skip
    # Let's be more comprehensive - include all plausible flower classes
    # and some plant-related ones
}

# Actually, let's use a broader approach: check class names
# ImageNet classes that contain flower-related words
FLOWER_KEYWORDS = [
    "daisy", "rose", "hip", "sunflower", "tulip", "orchid", "lily",
    "lotus", "poppy", "dandelion", "buttercup", "chamomile", "iris",
    "carnation", "lavender", "marigold", "petunia", "bouquet",
    "flower", "petal", "blossom", "floral",
    "bee", "butterfly",  # often on flowers
    "pot",  # flowerpot
    "vase",
    "bell pepper",  # flowering plant, close
    "mushroom",  # not flower but plant
    "coral fungus",
    "cardoon", "artichoke",  # flowering
]

def load_imagenet_classes():
    """Return dict of {index: class_name}."""
    # Use torchvision's built-in class list
    try:
        from torchvision.models import MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        categories = weights.meta["categories"]
        return {i: name for i, name in enumerate(categories)}
    except Exception:
        pass
    
    # Fallback: download the class list
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        r = requests.get(url, timeout=10)
        lines = r.text.strip().split("\n")
        return {i: name.strip() for i, name in enumerate(lines)}
    except Exception:
        return {}


def is_flower_class(class_name):
    """Check if an ImageNet class name corresponds to a flower."""
    name_lower = class_name.lower()
    
    # Explicit flower classes
    explicit_flowers = [
        "daisy", "yellow lady's slipper", "rose hip",
        "sunflower", "tulip", "orchid", "lily", "lotus",
        "poppy", "dandelion", "buttercup", "pot", "vase",
        "bee", "monarch", "lycaenid", "sulphur butterfly",
        "admiral", "ringlet", "cabbage butterfly",
        "hip, rose hip", "cardoon", "artichoke",
        "anemone", "coral fungus",
    ]
    
    for flower in explicit_flowers:
        if flower in name_lower:
            return True
    
    # Additional keywords
    for kw in ["flower", "petal", "blossom", "bouquet", "floral"]:
        if kw in name_lower:
            return True
    
    return False


def get_flower_class_indices(class_names):
    """Get all ImageNet class indices that are flowers."""
    flower_indices = set()
    for idx, name in class_names.items():
        if is_flower_class(name):
            flower_indices.add(idx)
    return flower_indices


def classify_image(model, transform, img_path, flower_indices, class_names, top_k=5):
    """
    Classify an image. Returns (is_flower, top_class_name, confidence).
    An image is considered a flower if any of its top-k predictions is a flower class.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        topk = torch.topk(probs, top_k)
        
        top_class = class_names.get(topk.indices[0].item(), "unknown")
        top_conf = topk.values[0].item()
        
        # Check if ANY of top-k predictions is a flower
        for i in range(top_k):
            idx = topk.indices[i].item()
            prob = topk.values[i].item()
            if idx in flower_indices and prob > 0.01:  # at least 1% confidence
                return True, top_class, top_conf
        
        return False, top_class, top_conf
    except Exception as e:
        return False, f"error: {e}", 0.0


def clean_folder(model, transform, flower_indices, class_names, folder_path, folder_label):
    """Remove non-flower images from a folder."""
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total = len(files)
    removed = 0
    kept = 0
    
    print(f"\n{'='*60}")
    print(f"Cleaning: {folder_label} ({total} images)")
    print(f"{'='*60}")
    
    for i, fname in enumerate(files):
        fpath = os.path.join(folder_path, fname)
        is_flower, top_class, conf = classify_image(model, transform, fpath, flower_indices, class_names)
        
        if not is_flower:
            os.remove(fpath)
            removed += 1
            print(f"  ❌ REMOVED: {fname} (predicted: {top_class}, conf: {conf:.3f})")
        else:
            kept += 1
            if (i + 1) % 50 == 0:
                print(f"  ✅ Progress: {i+1}/{total} checked, {kept} kept, {removed} removed")
    
    print(f"\n  Summary: {kept} kept, {removed} removed out of {total}")
    return kept


def download_flowers_from_tf_dataset():
    """
    Download the TensorFlow flowers dataset (clean, curated flower images).
    Contains: daisy, dandelion, roses, sunflowers, tulips
    These all go into other_flowers.
    """
    import tarfile
    
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    save_path = os.path.join(BASE_DIR, "flower_photos.tgz")
    extract_path = os.path.join(BASE_DIR, "flower_photos_tmp")
    
    print("\n📥 Downloading TensorFlow Flowers dataset (218 MB)...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (10 * 1024 * 1024) < 8192:
                    print(f"  Downloaded: {downloaded / (1024*1024):.0f} MB / {total / (1024*1024):.0f} MB")
        
        print("📦 Extracting...")
        with tarfile.open(save_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        
        return os.path.join(extract_path, "flower_photos")
    except Exception as e:
        print(f"❌ Failed to download TF flowers: {e}")
        return None


def download_lily_images_wiki():
    """Download lily-specific images from Wikipedia with better filtering."""
    HEADERS = {
        "User-Agent": "LilyClassificationResearch/1.0 (academic.project@uni.edu) requests/2.32"
    }
    
    # More specific lily species for better image quality
    lily_titles = [
        "Lilium", "Lilium_candidum", "Lilium_lancifolium",
        "Lilium_bulbiferum", "Lilium_martagon", "Lilium_regale",
        "Lilium_superbum", "Lilium_auratum", "Lilium_speciosum",
        "Lilium_longiflorum", "Lilium_philadelphicum",
        "Lilium_columbianum", "Lilium_pardalinum",
        "Lilium_formosanum", "Lilium_henryi", "Lilium_pumilum",
        "Lilium_davidii", "Lilium_hansonii", "Lilium_monadelphum",
        "Lilium_pyrenaicum", "Lilium_nepalense", "Lilium_mackliniae",
        "Lilium_leucanthum", "Lilium_washingtonianum",
        "Lilium_rubescens", "Lilium_humboldtii",
        "Lilium_michiganense", "Lilium_canadense",
        "Lilium_cernuum", "Lilium_duchartrei",
        "Nymphaea", "Nymphaea_alba", "Nymphaea_caerulea",
        "Nymphaea_lotus", "Nymphaea_nouchali",
        "Hemerocallis", "Hemerocallis_fulva",  # daylilies
    ]
    
    urls = []
    batch_size = 5
    for i in range(0, len(lily_titles), batch_size):
        batch = lily_titles[i:i+batch_size]
        title_str = "|".join(batch)
        api_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&generator=images"
            f"&titles={title_str}&gimlimit=500&prop=imageinfo&iiprop=url|mime&iiurlwidth=400&format=json"
        )
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for p in pages.values():
                    title = p.get("title", "").lower()
                    # Strong blacklist for non-flower content
                    blacklist = [
                        "icon", "logo", "flag", "map", "stub", "diagram",
                        "symbol", "drawing", "wikimedia", "button", "sign",
                        "badge", "distribution", "range", "commons-logo",
                        "wikispecies", "wikidata", "edit-clear", "folder",
                        "ambox", "question", "lock", "padlock", "info",
                        "crystal", "nuvola", "gnome", "tango", "ooui",
                        "status", "merge", "split", "redirect", "disambig",
                        "taxonomy", "cladogram", "phylo", "chromosome",
                        "karyotype", "seed", "fruit", "leaf", "bark",
                        "stem", "root", "bulb", "habitat", "landscape",
                        "garden", "field", "meadow", "people", "person",
                        "author", "portrait", "signature", "stamp",
                        "coin", "book", "text", "page", "scan",
                        "herbarium", "specimen", "pressed",
                        "coat of arms", "escudo", "wappen",
                    ]
                    if any(bad in title for bad in blacklist):
                        continue
                    # Must have flower-related keyword in title
                    flower_keywords = [
                        "lilium", "lily", "nymphaea", "hemerocallis",
                        "flower", "bloom", "blossom", "petal", "flora",
                        ".jpg", ".jpeg", ".png"
                    ]
                    ii = p.get("imageinfo", [])
                    if ii:
                        info = ii[0]
                        mime = info.get("mime", "").lower()
                        thumb_url = info.get("thumburl", "") or info.get("url", "")
                        if ("jpeg" in mime or "png" in mime) and thumb_url:
                            urls.append(thumb_url)
        except Exception as e:
            print(f"  Wiki batch error: {e}")
        time.sleep(0.3)
    
    return urls


def download_other_flowers_wiki():
    """Download other flower images from Wikipedia with better filtering."""
    HEADERS = {
        "User-Agent": "LilyClassificationResearch/1.0 (academic.project@uni.edu) requests/2.32"
    }
    
    other_titles = [
        "Rosa", "Rosa_gallica", "Rosa_chinensis", "Rosa_canina",
        "Rosa_damascena", "Rosa_rugosa",
        "Helianthus_annuus", "Helianthus",
        "Tulipa", "Tulipa_gesneriana", "Tulipa_clusiana",
        "Bellis_perennis", "Gerbera",
        "Tagetes", "Tagetes_erecta",
        "Hibiscus", "Hibiscus_rosa-sinensis",
        "Nelumbo_nucifera",
        "Phalaenopsis", "Orchidaceae",
        "Dahlia", "Dahlia_pinnata",
        "Lavandula", "Lavandula_angustifolia",
        "Papaver_rhoeas", "Papaver_somniferum",
        "Chrysanthemum", "Chrysanthemum_morifolium",
        "Iris_germanica", "Iris_pseudacorus",
        "Narcissus_(plant)", "Narcissus_pseudonarcissus",
        "Jasminum", "Jasminum_officinale",
        "Magnolia", "Magnolia_grandiflora",
        "Camellia", "Camellia_japonica",
        "Zinnia", "Zinnia_elegans",
        "Petunia",
        "Plumeria", "Plumeria_rubra",
        "Bougainvillea",
        "Viola_(plant)", "Viola_tricolor",
        "Antirrhinum", "Antirrhinum_majus",
        "Dianthus", "Dianthus_caryophyllus",
        "Cosmos_(plant)", "Cosmos_bipinnatus",
    ]
    
    urls = []
    batch_size = 5
    for i in range(0, len(other_titles), batch_size):
        batch = other_titles[i:i+batch_size]
        title_str = "|".join(batch)
        api_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&generator=images"
            f"&titles={title_str}&gimlimit=500&prop=imageinfo&iiprop=url|mime&iiurlwidth=400&format=json"
        )
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for p in pages.values():
                    title = p.get("title", "").lower()
                    blacklist = [
                        "icon", "logo", "flag", "map", "stub", "diagram",
                        "symbol", "drawing", "wikimedia", "button", "sign",
                        "badge", "distribution", "range", "commons-logo",
                        "wikispecies", "wikidata", "edit-clear", "folder",
                        "ambox", "question", "lock", "padlock", "info",
                        "crystal", "nuvola", "gnome", "tango", "ooui",
                        "status", "merge", "split", "redirect", "disambig",
                        "taxonomy", "cladogram", "phylo", "chromosome",
                        "karyotype", "seed", "fruit", "leaf", "bark",
                        "stem", "root", "bulb", "habitat", "landscape",
                        "garden", "field", "meadow", "people", "person",
                        "author", "portrait", "signature", "stamp",
                        "coin", "book", "text", "page", "scan",
                        "herbarium", "specimen", "pressed",
                        "coat of arms", "escudo", "wappen",
                    ]
                    if any(bad in title for bad in blacklist):
                        continue
                    ii = p.get("imageinfo", [])
                    if ii:
                        info = ii[0]
                        mime = info.get("mime", "").lower()
                        thumb_url = info.get("thumburl", "") or info.get("url", "")
                        if ("jpeg" in mime or "png" in mime) and thumb_url:
                            urls.append(thumb_url)
        except Exception as e:
            print(f"  Wiki batch error: {e}")
        time.sleep(0.3)
    
    return urls


def download_and_save(url, save_path, headers=None):
    """Download image, validate, resize and save."""
    if headers is None:
        headers = {"User-Agent": "LilyClassificationResearch/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return False
        img = Image.open(BytesIO(r.content))
        img = img.convert("RGB")
        if img.width < 80 or img.height < 80:
            return False
        img.thumbnail((300, 300))
        img.save(save_path, "JPEG", quality=85)
        return True
    except Exception:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        return False


def populate_from_tf_flowers(tf_dir, target_folder, target_count):
    """Copy flower images from TF flowers dataset to target folder."""
    if not tf_dir or not os.path.exists(tf_dir):
        return 0
    
    existing = len([f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    needed = target_count - existing
    if needed <= 0:
        return 0
    
    copied = 0
    # TF flowers has: daisy, dandelion, roses, sunflowers, tulips
    for subdir in sorted(os.listdir(tf_dir)):
        subpath = os.path.join(tf_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        if subdir.lower() == "license.txt":
            continue
        
        for fname in sorted(os.listdir(subpath)):
            if copied >= needed:
                break
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            src = os.path.join(subpath, fname)
            idx = existing + copied + 1
            dst = os.path.join(target_folder, f"other_flowers_{idx:04d}.jpg")
            
            try:
                img = Image.open(src).convert("RGB")
                img.thumbnail((300, 300))
                img.save(dst, "JPEG", quality=85)
                copied += 1
            except Exception:
                continue
        
        if copied >= needed:
            break
    
    print(f"  Copied {copied} images from TF flowers dataset")
    return copied


def main():
    TARGET_PER_CLASS = 520
    
    print("=" * 60)
    print("🌸 FLOWER DATASET CLEANER & REBUILDER")
    print("=" * 60)
    
    # ─── Step 1: Load pre-trained model ───
    print("\n📦 Loading pre-trained MobileNetV2...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = load_imagenet_classes()
    if not class_names:
        print("⚠️  Could not load ImageNet class names. Using index-based filtering.")
    else:
        print(f"  Loaded {len(class_names)} ImageNet class names")
    
    flower_indices = get_flower_class_indices(class_names)
    print(f"  Found {len(flower_indices)} flower-related classes in ImageNet")
    flower_class_list = [class_names[i] for i in sorted(flower_indices)]
    print(f"  Flower classes: {flower_class_list[:10]}...")
    
    # ─── Step 2: Clean existing images ───
    lily_dir = os.path.join(DATASET_DIR, "lily")
    other_dir = os.path.join(DATASET_DIR, "other_flowers")
    
    lily_count = clean_folder(model, transform, flower_indices, class_names, lily_dir, "lily")
    other_count = clean_folder(model, transform, flower_indices, class_names, other_dir, "other_flowers")
    
    print(f"\n📊 After cleaning:")
    print(f"  Lily: {lily_count} images")
    print(f"  Other flowers: {other_count} images")
    
    # ─── Step 3: Download TF flowers for other_flowers ───
    needed_other = TARGET_PER_CLASS - other_count
    needed_lily = TARGET_PER_CLASS - lily_count
    
    if needed_other > 0:
        print(f"\n📥 Need {needed_other} more other_flowers images")
        tf_dir = download_flowers_from_tf_dataset()
        if tf_dir:
            populate_from_tf_flowers(tf_dir, other_dir, TARGET_PER_CLASS)
            # Cleanup
            tgz = os.path.join(BASE_DIR, "flower_photos.tgz")
            tmp = os.path.join(BASE_DIR, "flower_photos_tmp")
            if os.path.exists(tgz):
                os.remove(tgz)
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
    
    # ─── Step 4: Download more lily images from Wiki ───
    lily_count = len([f for f in os.listdir(lily_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    needed_lily = TARGET_PER_CLASS - lily_count
    
    if needed_lily > 0:
        print(f"\n📥 Need {needed_lily} more lily images from Wikipedia...")
        wiki_urls = download_lily_images_wiki()
        print(f"  Found {len(wiki_urls)} candidate URLs")
        
        downloaded = 0
        for url in wiki_urls:
            if downloaded >= needed_lily:
                break
            idx = lily_count + downloaded + 1
            fpath = os.path.join(lily_dir, f"lily_{idx:04d}.jpg")
            if download_and_save(url, fpath):
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  Downloaded {downloaded}/{needed_lily} lily images")
            time.sleep(0.05)
        print(f"  Downloaded {downloaded} lily images from Wiki")
    
    # ─── Step 5: Download more other_flowers from Wiki if needed ───
    other_count = len([f for f in os.listdir(other_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    needed_other = TARGET_PER_CLASS - other_count
    
    if needed_other > 0:
        print(f"\n📥 Need {needed_other} more other_flowers from Wikipedia...")
        wiki_urls = download_other_flowers_wiki()
        print(f"  Found {len(wiki_urls)} candidate URLs")
        
        downloaded = 0
        for url in wiki_urls:
            if downloaded >= needed_other:
                break
            idx = other_count + downloaded + 1
            fpath = os.path.join(other_dir, f"other_flowers_{idx:04d}.jpg")
            if download_and_save(url, fpath):
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  Downloaded {downloaded}/{needed_other} other_flowers images")
            time.sleep(0.05)
        print(f"  Downloaded {downloaded} other_flowers from Wiki")
    
    # ─── Step 6: Re-verify all new downloads ───
    print("\n🔍 Re-verifying all images after download...")
    lily_count = clean_folder(model, transform, flower_indices, class_names, lily_dir, "lily (re-verify)")
    other_count = clean_folder(model, transform, flower_indices, class_names, other_dir, "other_flowers (re-verify)")
    
    # ─── Step 7: Renumber files cleanly ───
    for folder_name in ["lily", "other_flowers"]:
        folder = os.path.join(DATASET_DIR, folder_name)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # First rename to temp names to avoid conflicts
        for i, f in enumerate(files):
            src = os.path.join(folder, f)
            tmp = os.path.join(folder, f"_tmp_{i:04d}.jpg")
            os.rename(src, tmp)
        
        # Then rename to final names
        tmp_files = sorted([f for f in os.listdir(folder) if f.startswith("_tmp_")])
        for i, f in enumerate(tmp_files):
            src = os.path.join(folder, f)
            dst = os.path.join(folder, f"{folder_name}_{i+1:04d}.jpg")
            os.rename(src, dst)
    
    # ─── Final summary ───
    lily_final = len([f for f in os.listdir(lily_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    other_final = len([f for f in os.listdir(other_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  🌺 Lily flowers: {lily_final} images")
    print(f"  🌻 Other flowers: {other_final} images")
    print(f"  📁 Total: {lily_final + other_final} images")
    print(f"  📂 Location: {DATASET_DIR}")
    print(f"\n  All images are verified flower-only images! ✅")


if __name__ == "__main__":
    main()
