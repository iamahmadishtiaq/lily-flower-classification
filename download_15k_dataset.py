"""
Massive Scaled Dataset Builder: 15,000 Total Images (5,000 per class)
- dataset/lily: 5,000 verified Lilium + Dragon Lily flower photos
- dataset/other_flowers: 5,000 verified other flower photos (Roses, Tulips, Sunflowers, Daisies, Dandelions, Orchids, Hibiscus, Iris)
- dataset/not_a_flower: 5,000 verified non-flower photos (Mammals, Birds, Insects, Reptiles)
"""
import os
import sys
import shutil
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

BASE_DIR = "/Users/erbiumtech/Downloads/lily_classification"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TF_FLOWER_DIR = "/Users/erbiumtech/Downloads/flower_photos"
TARGET_PER_CLASS = 5000

def run_curl_json(url):
    cmd = ['curl', '-s', url, '-H', 'User-Agent: PureBotanyResearch/3.0 (academic; pk)']
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(res.stdout)
    except:
        return {}

def download_and_save(item):
    url, save_path = item
    tmp_path = save_path + ".tmp"
    cmd = ['curl', '-s', '--max-time', '6', url, '-o', tmp_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1200:
        try:
            with Image.open(tmp_path) as im:
                im = im.convert('RGB')
                if im.width >= 80 and im.height >= 80:
                    im = im.resize((224, 224), Image.Resampling.BILINEAR)
                    im.save(save_path, 'JPEG', quality=85)
                    os.remove(tmp_path)
                    return True
        except:
            pass
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False

# ════════════════════════════════════════════════════════════
# 1. LILY FOLDER (5,000 PURE LILIES)
# ════════════════════════════════════════════════════════════
def build_lily_folder():
    folder = os.path.join(DATASET_DIR, "lily")
    print(f"\n🌺 [1/3] Building 5,000 PURE LILY folder: {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    urls = []

    # Dragon lilies
    print("  Fetching Dragon Lily photo URLs...")
    d_data = run_curl_json("https://api.inaturalist.org/v1/observations?taxon_name=Dracunculus+vulgaris&quality_grade=research&photos=true&per_page=100")
    for obs in d_data.get('results', []):
        for p in obs.get('photos', []):
            u = p.get('url', '').replace('square', 'medium')
            if u:
                urls.append(u)
    print(f"  Got {len(urls)} Dragon Lily URLs.")

    # Lilium genus
    for page in range(1, 25):
        print(f"  Fetching Lilium genus observations (page {page}/24)...")
        data = run_curl_json(f"https://api.inaturalist.org/v1/observations?taxon_name=Lilium&quality_grade=research&photos=true&per_page=200&page={page}")
        for obs in data.get('results', []):
            for p in obs.get('photos', []):
                u = p.get('url', '').replace('square', 'medium')
                if u and u not in urls:
                    urls.append(u)
        print(f"    Total candidate lily URLs so far: {len(urls)}")
        if len(urls) >= TARGET_PER_CLASS + 500:
            break

    print(f"  Starting parallel download of 5,000 Lilies with 32 workers...")
    tasks = [(url, os.path.join(folder, f"lily_{i+1:05d}.jpg")) for i, url in enumerate(urls[:TARGET_PER_CLASS + 600])]

    saved = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download_and_save, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                saved += 1
                if saved % 250 == 0:
                    rate = saved / (time.time() - t0)
                    print(f"    Saved: {saved}/{TARGET_PER_CLASS} Lilies ({rate:.1f} img/s)")
                if saved >= TARGET_PER_CLASS:
                    break

    # Renumber sequentially
    files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])[:TARGET_PER_CLASS]
    for i, f in enumerate(files):
        src = os.path.join(folder, f)
        dst = os.path.join(folder, f"lily_{i+1:05d}.jpg")
        if src != dst:
            os.rename(src, dst)
    all_f = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for extra in all_f[TARGET_PER_CLASS:]:
        os.remove(os.path.join(folder, extra))

    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} pure Lilies!")
    return actual

# ════════════════════════════════════════════════════════════
# 2. OTHER FLOWERS FOLDER (5,000 PURE OTHER FLOWERS)
# ════════════════════════════════════════════════════════════
def build_other_flowers_folder():
    folder = os.path.join(DATASET_DIR, "other_flowers")
    print(f"\n🌻 [2/3] Building 5,000 PURE OTHER FLOWERS folder: {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    saved = 0
    # Copy all from TF flowers first (3,670 images)
    if os.path.exists(TF_FLOWER_DIR):
        print("  Importing curated TensorFlow flowers (Roses, Tulips, Sunflowers, Daisies, Dandelions)...")
        for cat in ["roses", "tulips", "sunflowers", "daisy", "dandelion"]:
            cat_dir = os.path.join(TF_FLOWER_DIR, cat)
            if not os.path.exists(cat_dir):
                continue
            cat_saved = 0
            for f in sorted(os.listdir(cat_dir)):
                if saved >= TARGET_PER_CLASS:
                    break
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(cat_dir, f)
                    dst = os.path.join(folder, f"other_flowers_{saved+1:05d}.jpg")
                    try:
                        with Image.open(src) as im:
                            im = im.convert('RGB')
                            im = im.resize((224, 224), Image.Resampling.BILINEAR)
                            im.save(dst, 'JPEG', quality=85)
                            saved += 1
                            cat_saved += 1
                    except:
                        pass
            print(f"    Copied {cat_saved} photos from {cat} (Total: {saved})")

    # Top up remaining to reach 5,000 from iNaturalist (Orchidaceae, Hibiscus, Iris, Papaver)
    needed = TARGET_PER_CLASS - saved
    if needed > 0:
        print(f"  Fetching {needed} additional flower URLs from iNaturalist (Orchids, Hibiscus, Iris, Poppies)...")
        extra_urls = []
        for taxon in ["Orchidaceae", "Hibiscus", "Iris", "Papaver"]:
            for page in (1, 2):
                data = run_curl_json(f"https://api.inaturalist.org/v1/observations?taxon_name={taxon}&quality_grade=research&photos=true&per_page=150&page={page}")
                for obs in data.get('results', []):
                    for p in obs.get('photos', []):
                        u = p.get('url', '').replace('square', 'medium')
                        if u and u not in extra_urls:
                            extra_urls.append(u)
                if len(extra_urls) >= needed + 300:
                    break

        print(f"  Downloading {needed} top-up flower photos with 32 workers...")
        tasks = [(url, os.path.join(folder, f"other_flowers_{saved+i+1:05d}.jpg")) for i, url in enumerate(extra_urls[:needed + 400])]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(download_and_save, t) for t in tasks]
            for f in as_completed(futures):
                if f.result():
                    saved += 1
                    if saved % 250 == 0:
                        print(f"    Saved: {saved}/{TARGET_PER_CLASS} other flowers")
                    if saved >= TARGET_PER_CLASS:
                        break

    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} pure other flowers!")
    return actual

# ════════════════════════════════════════════════════════════
# 3. NOT A FLOWER FOLDER (5,000 PURE NON-FLOWER OBJECTS)
# ════════════════════════════════════════════════════════════
def build_not_a_flower_folder():
    folder = os.path.join(DATASET_DIR, "not_a_flower")
    print(f"\n🚫 [3/3] Building 5,000 PURE NOT-A-FLOWER folder: {folder}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    urls = []
    taxa_list = ["Mammalia", "Aves", "Insecta", "Reptilia"]
    per_taxa = TARGET_PER_CLASS // len(taxa_list) + 150

    for taxa in taxa_list:
        print(f"  Fetching non-flower observations ({taxa})...")
        taxa_urls = []
        for page in range(1, 8):
            data = run_curl_json(f"https://api.inaturalist.org/v1/observations?iconic_taxa={taxa}&quality_grade=research&photos=true&per_page=200&page={page}")
            for obs in data.get('results', []):
                for p in obs.get('photos', []):
                    u = p.get('url', '').replace('square', 'medium')
                    if u and u not in taxa_urls:
                        taxa_urls.append(u)
            if len(taxa_urls) >= per_taxa:
                break
        print(f"    Got {len(taxa_urls)} candidate URLs for {taxa}")
        urls.extend(taxa_urls[:per_taxa])

    print(f"  Total {len(urls)} candidate non-flower URLs. Starting parallel download...")
    tasks = [(url, os.path.join(folder, f"not_flower_{i+1:05d}.jpg")) for i, url in enumerate(urls[:TARGET_PER_CLASS + 600])]

    saved = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download_and_save, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                saved += 1
                if saved % 250 == 0:
                    rate = saved / (time.time() - t0)
                    print(f"    Saved: {saved}/{TARGET_PER_CLASS} non-flowers ({rate:.1f} img/s)")
                if saved >= TARGET_PER_CLASS:
                    break

    # Renumber sequentially
    files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])[:TARGET_PER_CLASS]
    for i, f in enumerate(files):
        src = os.path.join(folder, f)
        dst = os.path.join(folder, f"not_flower_{i+1:05d}.jpg")
        if src != dst:
            os.rename(src, dst)
    all_f = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for extra in all_f[TARGET_PER_CLASS:]:
        os.remove(os.path.join(folder, extra))

    actual = len(os.listdir(folder))
    print(f"  ✅ Completed {folder}: {actual} non-flower photos!")
    return actual

def main():
    print("="*60)
    print("🚀 BUILDING 15,000 VERIFIED DATASET (5,000 PER CLASS)")
    print("="*60)
    t_start = time.time()

    l_cnt = build_lily_folder()
    o_cnt = build_other_flowers_folder()
    n_cnt = build_not_a_flower_folder()

    t_total = time.time() - t_start
    print("\n" + "="*60)
    print("🎯 FINAL 15,000 DATASET SUMMARY:")
    print(f"  🌺 Lily flowers       : {l_cnt} (All genuine Lilium / Dragon Lilies)")
    print(f"  🌻 Other flowers      : {o_cnt} (All genuine non-lily flowers)")
    print(f"  🚫 Not a flower       : {n_cnt} (All genuine non-flower objects)")
    print(f"  📊 Total              : {l_cnt + o_cnt + n_cnt} images")
    print(f"  ⏱️ Time taken         : {t_total // 60:.0f}m {t_total % 60:.0f}s")
    print("="*60)

if __name__ == "__main__":
    main()
