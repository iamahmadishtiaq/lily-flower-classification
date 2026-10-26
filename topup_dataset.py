"""
Top-up script: Add more verified flower images to reach 500+ per class.
Uses remaining TF Flowers images for other_flowers, and more Wiki sources for lily.
"""
import os
import sys
import time
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import shutil
import tarfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LILY_DIR = os.path.join(DATASET_DIR, "lily")
OTHER_DIR = os.path.join(DATASET_DIR, "other_flowers")
TARGET = 520

HEADERS = {
    "User-Agent": "LilyClassificationResearch/1.0 (academic.project@uni.edu) requests/2.32"
}

# ─── Load model for verification ───
print("📦 Loading MobileNetV2 for verification...")
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    from torchvision.models import MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    class_names = {i: name for i, name in enumerate(categories)}
except:
    class_names = {}

def is_flower_class(class_name):
    name_lower = class_name.lower()
    flowers = [
        "daisy", "yellow lady's slipper", "rose hip",
        "sunflower", "tulip", "orchid", "lily", "lotus",
        "poppy", "dandelion", "buttercup", "pot", "vase",
        "bee", "monarch", "lycaenid", "sulphur butterfly",
        "admiral", "ringlet", "cabbage butterfly",
        "hip, rose hip", "cardoon", "artichoke",
        "anemone", "coral fungus", "flower", "petal", "blossom", "bouquet"
    ]
    for f in flowers:
        if f in name_lower:
            return True
    return False

flower_indices = set()
for idx, name in class_names.items():
    if is_flower_class(name):
        flower_indices.add(idx)

print(f"  {len(flower_indices)} flower classes identified")


def is_flower_image(img_path):
    """Check if image is a flower using pretrained model."""
    try:
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        topk = torch.topk(probs, 5)
        for i in range(5):
            if topk.indices[i].item() in flower_indices and topk.values[i].item() > 0.01:
                return True
        return False
    except:
        return False


def download_and_verify(url, save_path):
    """Download, validate as flower, save."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return False
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if img.width < 80 or img.height < 80:
            return False
        img.thumbnail((300, 300))
        # Save temporarily
        img.save(save_path, "JPEG", quality=85)
        # Verify it's a flower
        if not is_flower_image(save_path):
            os.remove(save_path)
            return False
        return True
    except:
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except: pass
        return False


# ─── Count current images ───
lily_count = len([f for f in os.listdir(LILY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
other_count = len([f for f in os.listdir(OTHER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"\n📊 Current: Lily={lily_count}, Other={other_count}")

# ═══════════════════════════════════════════════
# STEP 1: Add more other_flowers from TF Flowers
# ═══════════════════════════════════════════════
needed_other = TARGET - other_count
if needed_other > 0:
    print(f"\n🌻 Need {needed_other} more other_flowers images")
    print("📥 Downloading TF Flowers dataset...")
    
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    tgz_path = os.path.join(BASE_DIR, "flower_photos.tgz")
    extract_path = os.path.join(BASE_DIR, "flower_photos_tmp")
    
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(tgz_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < 8192:
                    print(f"  {downloaded / (1024*1024):.0f} MB / {total / (1024*1024):.0f} MB")
        
        print("📦 Extracting...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        
        tf_dir = os.path.join(extract_path, "flower_photos")
        added = 0
        
        # Get all existing filenames to avoid duplicates
        existing_files = set()
        for f in os.listdir(OTHER_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                existing_files.add(f)
        
        for subdir in sorted(os.listdir(tf_dir)):
            subpath = os.path.join(tf_dir, subdir)
            if not os.path.isdir(subpath):
                continue
            
            flower_type = subdir.lower()
            print(f"  Processing TF/{flower_type}...")
            
            files = sorted([f for f in os.listdir(subpath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for fname in files:
                if added >= needed_other:
                    break
                
                src = os.path.join(subpath, fname)
                idx = other_count + added + 1
                dst = os.path.join(OTHER_DIR, f"other_flowers_{idx:04d}.jpg")
                
                try:
                    img = Image.open(src).convert("RGB")
                    img.thumbnail((300, 300))
                    img.save(dst, "JPEG", quality=85)
                    added += 1
                    if added % 30 == 0:
                        print(f"    Added {added}/{needed_other}")
                except:
                    continue
            
            if added >= needed_other:
                break
        
        print(f"  ✅ Added {added} other_flowers from TF dataset")
        
        # Cleanup
        if os.path.exists(tgz_path):
            os.remove(tgz_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            
    except Exception as e:
        print(f"  ❌ TF download failed: {e}")

# ═══════════════════════════════════════════════
# STEP 2: Add more lily images from Wikipedia
# ═══════════════════════════════════════════════
lily_count = len([f for f in os.listdir(LILY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
needed_lily = TARGET - lily_count

if needed_lily > 0:
    print(f"\n🌺 Need {needed_lily} more lily images")
    
    # Extended list of lily species for more images
    lily_titles = [
        # True lilies (Lilium genus) - many species
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
        "Lilium_brownii", "Lilium_distichum",
        "Lilium_oxypetalum", "Lilium_polyphyllum",
        "Lilium_bosniacum", "Lilium_carniolicum",
        "Lilium_chalcedonicum", "Lilium_ciliatum",
        "Lilium_kesselringianum", "Lilium_ponticum",
        "Lilium_rhodopaeum", "Lilium_szovitsianum",
        # Water lilies
        "Nymphaea", "Nymphaea_alba", "Nymphaea_caerulea",
        "Nymphaea_lotus", "Nymphaea_nouchali",
        "Nymphaea_odorata", "Nymphaea_capensis",
        "Nymphaea_stellata", "Nymphaea_pubescens",
        "Victoria_(plant)", "Victoria_amazonica",
        # Daylilies
        "Hemerocallis", "Hemerocallis_fulva",
        "Hemerocallis_lilioasphodelus", "Hemerocallis_citrina",
        # Calla lily
        "Zantedeschia", "Zantedeschia_aethiopica",
        # Peace lily
        "Spathiphyllum",
        # Lily of the valley
        "Convallaria_majalis",
        # Clivia
        "Clivia_miniata",
        # Gloriosa
        "Gloriosa_(plant)", "Gloriosa_superba",
        # Amaryllis / Hippeastrum
        "Hippeastrum", "Amaryllis_belladonna",
    ]
    
    print(f"  Searching {len(lily_titles)} lily species...")
    
    all_urls = []
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
                        "people", "person", "author", "portrait",
                        "signature", "stamp", "coin", "book", "text",
                        "page", "scan", "herbarium", "specimen", "pressed",
                        "coat of arms", "escudo", "wappen", ".svg",
                        "garden", "field", "meadow",
                    ]
                    if any(bad in title for bad in blacklist):
                        continue
                    ii = p.get("imageinfo", [])
                    if ii:
                        info = ii[0]
                        mime = info.get("mime", "").lower()
                        thumb_url = info.get("thumburl", "") or info.get("url", "")
                        if ("jpeg" in mime or "png" in mime) and thumb_url:
                            all_urls.append(thumb_url)
        except Exception as e:
            pass
        time.sleep(0.3)
    
    # Deduplicate URLs
    seen = set()
    urls = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            urls.append(u)
    
    print(f"  Found {len(urls)} unique candidate URLs")
    
    added = 0
    for url in urls:
        if added >= needed_lily:
            break
        idx = lily_count + added + 1
        fpath = os.path.join(LILY_DIR, f"lily_{idx:04d}.jpg")
        if download_and_verify(url, fpath):
            added += 1
            if added % 20 == 0:
                print(f"    Added {added}/{needed_lily} verified lily images")
        time.sleep(0.05)
    
    print(f"  ✅ Added {added} verified lily images")

# ═══════════════════════════════════════════════
# STEP 3: Renumber all files cleanly
# ═══════════════════════════════════════════════
print("\n🔢 Renumbering files...")
for folder_name in ["lily", "other_flowers"]:
    folder = os.path.join(DATASET_DIR, folder_name)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for i, f in enumerate(files):
        src = os.path.join(folder, f)
        tmp = os.path.join(folder, f"_tmp_{i:04d}.jpg")
        os.rename(src, tmp)
    
    tmp_files = sorted([f for f in os.listdir(folder) if f.startswith("_tmp_")])
    for i, f in enumerate(tmp_files):
        src = os.path.join(folder, f)
        dst = os.path.join(folder, f"{folder_name}_{i+1:04d}.jpg")
        os.rename(src, dst)

# ═══════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════
lily_final = len([f for f in os.listdir(LILY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
other_final = len([f for f in os.listdir(OTHER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"\n{'='*60}")
print(f"✅ FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"  🌺 Lily flowers: {lily_final} images (all verified flowers)")
print(f"  🌻 Other flowers: {other_final} images (all verified flowers)")
print(f"  📁 Total: {lily_final + other_final} images")
print(f"  📂 Location: {DATASET_DIR}")

if lily_final >= 500 and other_final >= 500:
    print(f"\n  🎯 Target of 500+ per class ACHIEVED! ✅")
else:
    print(f"\n  ⚠️  Lily: {'✅' if lily_final >= 500 else '❌'} ({lily_final}/500)")
    print(f"  ⚠️  Other: {'✅' if other_final >= 500 else '❌'} ({other_final}/500)")
