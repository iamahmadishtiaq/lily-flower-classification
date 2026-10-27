"""
Fast Multi-threaded Builder for 100% Pure 3-Class Botanical Dataset:
1. dataset/lily: 520 Research-grade Lilium + Dragon Lily verified flower photos
2. dataset/other_flowers: 520 Verified non-lily flower photos from TF Flowers (roses, tulips, sunflowers, daisies, dandelions)
3. dataset/not_a_flower: 520 Verified non-flower photos (mammals, birds, insects, reptiles)
"""
import os
import sys
import shutil
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    except:
        return {}

def download_and_validate(item):
    url, save_path = item
    tmp_path = save_path + ".tmp"
    cmd = ['curl', '-s', '--max-time', '6', url, '-o', tmp_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1500:
        try:
            with Image.open(tmp_path) as im:
                im = im.convert('RGB')
                if im.width >= 100 and im.height >= 100:
                    im.thumbnail((300, 300))
                    im.save(save_path, 'JPEG', quality=88)
                    os.remove(tmp_path)
                    return True
        except:
            pass
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False

# 1. PURE LILIES
def build_lily_folder():
    folder = os.path.join(DATASET_DIR, "lily")
    print(f"\n🌺 Building 100% PURE LILY folder: {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    urls = []
    
    # Dragon lilies
    print("  Fetching Dragon Lily photo URLs...")
    d_data = run_curl_json("https://api.inaturalist.org/v1/observations?taxon_name=Dracunculus+vulgaris&quality_grade=research&photos=true&per_page=30")
    for obs in d_data.get('results', []):
        for p in obs.get('photos', []):
            u = p.get('url', '').replace('square', 'medium')
            if u:
                urls.append(u)
    print(f"  Got {len(urls)} Dragon Lily URLs.")
    
    # Lilium genus
    for page in range(1, 6):
        print(f"  Fetching Lilium genus URLs (page {page})...")
        data = run_curl_json(f"https://api.inaturalist.org/v1/observations?taxon_name=Lilium&quality_grade=research&photos=true&per_page=100&page={page}")
        for obs in data.get('results', []):
            for p in obs.get('photos', []):
                u = p.get('url', '').replace('square', 'medium')
                if u and u not in urls:
                    urls.append(u)
        if len(urls) >= TARGET_COUNT + 100:
            break
    
    print(f"  Total {len(urls)} candidate URLs collected. Starting parallel download...")
    tasks = [(url, os.path.join(folder, f"lily_{i+1:04d}.jpg")) for i, url in enumerate(urls[:TARGET_COUNT + 80])]
    
    saved = 0
    with ThreadPoolExecutor(max_workers=14) as pool:
        futures = [pool.submit(download_and_validate, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                saved += 1
                if saved % 50 == 0:
                    print(f"    Saved: {saved}/{TARGET_COUNT} pure lilies")
                if saved >= TARGET_COUNT:
                    break
    
    # Renumber sequentially
    files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])[:TARGET_COUNT]
    for i, f in enumerate(files):
        src = os.path.join(folder, f)
        dst = os.path.join(folder, f"lily_{i+1:04d}.jpg")
        if src != dst:
            os.rename(src, dst)
    # Remove extras if any
    all_f = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for extra in all_f[TARGET_COUNT:]:
        os.remove(os.path.join(folder, extra))
    
    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} pure Lilies!")
    return actual

# 2. PURE OTHER FLOWERS
def build_other_flowers_folder():
    folder = os.path.join(DATASET_DIR, "other_flowers")
    print(f"\n🌻 Building 100% PURE OTHER FLOWERS folder: {folder}")
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
    
    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} pure other flowers!")
    return actual

# 3. PURE NOT A FLOWER OBJECTS
def build_not_a_flower_folder():
    folder = os.path.join(DATASET_DIR, "not_a_flower")
    print(f"\n🚫 Building 100% PURE NOT-A-FLOWER folder: {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
    urls = []
    taxa_list = ["Mammalia", "Aves", "Insecta", "Reptilia"]
    for taxa in taxa_list:
        print(f"  Fetching {taxa} URLs...")
        for page in (1, 2):
            data = run_curl_json(f"https://api.inaturalist.org/v1/observations?iconic_taxa={taxa}&quality_grade=research&photos=true&per_page=80&page={page}")
            for obs in data.get('results', []):
                for p in obs.get('photos', []):
                    u = p.get('url', '').replace('square', 'medium')
                    if u and u not in urls:
                        urls.append(u)
    
    print(f"  Total {len(urls)} candidate non-flower URLs collected. Starting parallel download...")
    tasks = [(url, os.path.join(folder, f"not_flower_{i+1:04d}.jpg")) for i, url in enumerate(urls[:TARGET_COUNT + 80])]
    
    saved = 0
    with ThreadPoolExecutor(max_workers=14) as pool:
        futures = [pool.submit(download_and_validate, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                saved += 1
                if saved % 50 == 0:
                    print(f"    Saved: {saved}/{TARGET_COUNT} non-flowers")
                if saved >= TARGET_COUNT:
                    break
    
    # Renumber sequentially
    files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])[:TARGET_COUNT]
    for i, f in enumerate(files):
        src = os.path.join(folder, f)
        dst = os.path.join(folder, f"not_flower_{i+1:04d}.jpg")
        if src != dst:
            os.rename(src, dst)
    # Remove extras if any
    all_f = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for extra in all_f[TARGET_COUNT:]:
        os.remove(os.path.join(folder, extra))
    
    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} non-flower photos!")
    return actual

def main():
    print("="*60)
    print("🚀 FAST PARALLEL DATASET BUILDER (3 CLASSES)")
    print("="*60)
    
    l_cnt = build_lily_folder()
    o_cnt = build_other_flowers_folder()
    n_cnt = build_not_a_flower_folder()
    
    print("\n" + "="*60)
    print("🎯 FINAL DATASET SUMMARY:")
    print(f"  🌺 Lily flowers       : {l_cnt}")
    print(f"  🌻 Other flowers      : {o_cnt}")
    print(f"  🚫 Not a flower       : {n_cnt}")
    print(f"  📊 Total              : {l_cnt + o_cnt + n_cnt} images")
    print("="*60)

if __name__ == "__main__":
    main()
