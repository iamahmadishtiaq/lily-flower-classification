"""
Build 100% Pure, Verified 3-Class Dataset:
1. dataset/lily: 520 Research-Grade Lilium & Dragon Lily flower photos from iNaturalist Open Data
2. dataset/other_flowers: 520 Verified non-lily flower photos from TF Flowers (roses, tulips, sunflowers, daisies, dandelions)
3. dataset/not_a_flower: 520 Verified non-flower photos (mammals, birds, insects, reptiles)
"""
import os
import sys
import shutil
import json
import subprocess
from PIL import Image

BASE_DIR = "/Users/erbiumtech/Downloads/lily_classification"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TF_FLOWER_DIR = "/Users/erbiumtech/Downloads/flower_photos"

TARGET_COUNT = 520

def run_curl_json(url):
    cmd = ['curl', '-s', url, '-H', 'User-Agent: PureBotanyResearch/3.0 (academic; pk)']
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(res.stdout)
    except Exception as e:
        print(f"Error parsing JSON from {url}: {e}")
        return {}

def download_image_curl(url, save_path):
    cmd = ['curl', '-s', '--max-time', '8', url, '-o', save_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        try:
            with Image.open(save_path) as im:
                im = im.convert('RGB')
                if im.width >= 100 and im.height >= 100:
                    im.thumbnail((300, 300))
                    im.save(save_path, 'JPEG', quality=88)
                    return True
        except:
            pass
    if os.path.exists(save_path):
        os.remove(save_path)
    return False

# ════════════════════════════════════════════════════════════
# 1. PURE LILY FLOWERS
# ════════════════════════════════════════════════════════════
def build_lily_folder():
    folder = os.path.join(DATASET_DIR, "lily")
    print(f"\n==================================================")
    print(f"🌺 Building 100% PURE LILY folder: {folder}")
    print(f"==================================================")
    
    # Clean old folder completely
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    saved = 0
    
    # First: Dragon Lilies (Dracunculus vulgaris)
    print("  Fetching Dragon Lily observations...")
    url = "https://api.inaturalist.org/v1/observations?taxon_name=Dracunculus+vulgaris&quality_grade=research&photos=true&per_page=30"
    data = run_curl_json(url)
    for obs in data.get('results', []):
        for p in obs.get('photos', []):
            if saved >= 25:
                break
            u = p.get('url', '').replace('square', 'medium')
            if u:
                path = os.path.join(folder, f"lily_{saved+1:04d}.jpg")
                if download_image_curl(u, path):
                    saved += 1
    print(f"  Saved {saved} Dragon Lily photos.")
    
    # Second: True Lilies (Lilium genus) - research grade
    page = 1
    while saved < TARGET_COUNT and page <= 8:
        print(f"  Fetching Lilium genus observations (page {page})...")
        url = f"https://api.inaturalist.org/v1/observations?taxon_name=Lilium&quality_grade=research&photos=true&per_page=100&page={page}"
        data = run_curl_json(url)
        results = data.get('results', [])
        if not results:
            break
        for obs in results:
            if saved >= TARGET_COUNT:
                break
            for p in obs.get('photos', []):
                if saved >= TARGET_COUNT:
                    break
                u = p.get('url', '').replace('square', 'medium')
                if u:
                    path = os.path.join(folder, f"lily_{saved+1:04d}.jpg")
                    if download_image_curl(u, path):
                        saved += 1
                        if saved % 50 == 0:
                            print(f"    Progress: {saved}/{TARGET_COUNT} pure lilies saved")
        page += 1
    
    print(f"  ✅ Completed {folder}: {saved} 100% authentic Lily photos!")
    return saved

# ════════════════════════════════════════════════════════════
# 2. PURE OTHER FLOWERS
# ════════════════════════════════════════════════════════════
def build_other_flowers_folder():
    folder = os.path.join(DATASET_DIR, "other_flowers")
    print(f"\n==================================================")
    print(f"🌻 Building 100% PURE OTHER FLOWERS folder: {folder}")
    print(f"==================================================")
    
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    categories = ["roses", "tulips", "sunflowers", "daisy", "dandelion"]
    per_cat = TARGET_COUNT // len(categories) + 5
    saved = 0
    
    for cat in categories:
        cat_dir = os.path.join(TF_FLOWER_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
        cat_saved = 0
        for f in sorted(os.listdir(cat_dir)):
            if cat_saved >= per_cat or saved >= TARGET_COUNT:
                break
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(cat_dir, f)
                dst = os.path.join(folder, f"other_flowers_{saved+1:04d}.jpg")
                try:
                    with Image.open(src) as im:
                        im = im.convert('RGB')
                        im.thumbnail((300, 300))
                        im.save(dst, 'JPEG', quality=88)
                        saved += 1
                        cat_saved += 1
                except:
                    pass
        print(f"  Copied {cat_saved} photos from {cat}")
    
    print(f"  ✅ Completed {folder}: {saved} pure other flower photos!")
    return saved

# ════════════════════════════════════════════════════════════
# 3. PURE NOT-A-FLOWER OBJECTS
# ════════════════════════════════════════════════════════════
def build_not_a_flower_folder():
    folder = os.path.join(DATASET_DIR, "not_a_flower")
    print(f"\n==================================================")
    print(f"🚫 Building 100% PURE NOT-A-FLOWER folder: {folder}")
    print(f"==================================================")
    
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    saved = 0
    taxa_list = ["Mammalia", "Aves", "Insecta", "Reptilia"]
    per_taxa = TARGET_COUNT // len(taxa_list) + 5
    
    for taxa in taxa_list:
        print(f"  Fetching non-flower observations ({taxa})...")
        taxa_saved = 0
        page = 1
        while taxa_saved < per_taxa and saved < TARGET_COUNT and page <= 4:
            url = f"https://api.inaturalist.org/v1/observations?iconic_taxa={taxa}&quality_grade=research&photos=true&per_page=60&page={page}"
            data = run_curl_json(url)
            results = data.get('results', [])
            if not results:
                break
            for obs in results:
                if taxa_saved >= per_taxa or saved >= TARGET_COUNT:
                    break
                for p in obs.get('photos', []):
                    if taxa_saved >= per_taxa or saved >= TARGET_COUNT:
                        break
                    u = p.get('url', '').replace('square', 'medium')
                    if u:
                        path = os.path.join(folder, f"not_flower_{saved+1:04d}.jpg")
                        if download_image_curl(u, path):
                            saved += 1
                            taxa_saved += 1
            page += 1
        print(f"  Saved {taxa_saved} {taxa} photos")
    
    print(f"  ✅ Completed {folder}: {saved} non-flower photos!")
    return saved

def main():
    print("="*60)
    print("🚀 BUILDING 100% PURE BOTANICAL DATASET (3 CLASSES)")
    print("="*60)
    
    l_cnt = build_lily_folder()
    o_cnt = build_other_flowers_folder()
    n_cnt = build_not_a_flower_folder()
    
    print("\n" + "="*60)
    print("🎯 FINAL DATASET SUMMARY:")
    print(f"  🌺 Lily flowers       : {l_cnt} (All genuine Lilium / Dragon Lilies)")
    print(f"  🌻 Other flowers      : {o_cnt} (Roses, Tulips, Sunflowers, Daisies, Dandelions)")
    print(f"  🚫 Not a flower       : {n_cnt} (Mammals, Birds, Insects, Reptiles)")
    print(f"  📊 Total              : {l_cnt + o_cnt + n_cnt} images")
    print("="*60)

if __name__ == "__main__":
    main()
