import os
import sys
import time
import requests
from PIL import Image
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

HEADERS = {
    "User-Agent": "LilyClassificationResearch/1.0 (academic.project@uni.edu) requests/2.32"
}

CATEGORIES = {
    "lily": [
        "Lilium",
        "Lilium_candidum",
        "Lilium_lancifolium",
        "Lilium_bulbiferum",
        "Lilium_martagon",
        "Lilium_regale",
        "Lilium_superbum",
        "Lilium_auratum",
        "Lilium_speciosum",
        "Lilium_longiflorum",
        "Lilium_philadelphicum",
        "Lilium_columbianum",
        "Lilium_pardalinum",
        "Nymphaea_alba",
        "Nymphaea_caerulea",
        "Nymphaea_lotus",
        "Spathiphyllum"
    ],
    "other_flowers": [
        "Rosa",
        "Rose",
        "Rosa_gallica",
        "Rosa_chinensis",
        "Rosa_canina",
        "Helianthus_annuus",
        "Tulipa",
        "Tulipa_gesneriana",
        "Bellis_perennis",
        "Tagetes",
        "Tagetes_erecta",
        "Taraxacum_officinale",
        "Hibiscus_syriacus",
        "Nelumbo_nucifera",
        "Phalaenopsis",
        "Dahlia",
        "Lavandula_angustifolia",
        "Papaver_rhoeas",
        "Chrysanthemum",
        "Iris_(plant)"
    ]
}

BLACKLIST_KEYWORDS = [
    "icon", "logo", "flag", "map", "stub", "diagram", "symbol", 
    "drawing", "wikimedia", "button", "sign", "badge", "distribution"
]

def fetch_image_urls_for_titles(titles):
    urls = []
    batch_size = 8
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        title_str = "|".join(batch)
        api_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&generator=images"
            f"&titles={title_str}&gimlimit=250&prop=imageinfo&iiprop=url|mime&iiurlwidth=400&format=json"
        )
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for p in pages.values():
                    title = p.get("title", "").lower()
                    if any(bad in title for bad in BLACKLIST_KEYWORDS):
                        continue
                    ii = p.get("imageinfo", [])
                    if ii:
                        info = ii[0]
                        mime = info.get("mime", "").lower()
                        thumb_url = info.get("thumburl", "") or info.get("url", "")
                        
                        if ("jpeg" in mime or "png" in mime or "jpg" in thumb_url or "jpeg" in thumb_url or "png" in thumb_url) and thumb_url:
                            urls.append(thumb_url)
        except Exception as e:
            print(f"Error fetching batch {batch}: {e}")
        time.sleep(0.2)
    return urls

def download_and_save(url, save_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return False
        
        img = Image.open(BytesIO(r.content))
        img = img.convert("RGB")
        if img.width < 100 or img.height < 100:
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

def collect_dataset(target_per_class=100):
    for class_name, titles in CATEGORIES.items():
        folder = os.path.join(DATASET_DIR, class_name)
        os.makedirs(folder, exist_ok=True)
        
        existing = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(existing)
        print(f"\n================ Class: {class_name} ================")
        print(f"Existing valid images: {count}")
        
        if count >= target_per_class:
            print(f"Already reached target of {target_per_class} images. Skipping.")
            continue
            
        print(f"Fetching URLs for {len(titles)} encyclopedia topics...")
        urls = fetch_image_urls_for_titles(titles)
        print(f"Found {len(urls)} candidate URLs.")
        
        for url in urls:
            if count >= target_per_class:
                break
            filename = f"{class_name}_{count+1:04d}.jpg"
            filepath = os.path.join(folder, filename)
            if download_and_save(url, filepath):
                count += 1
                if count % 15 == 0 or count == target_per_class:
                    print(f"  [{class_name}] Progress: {count}/{target_per_class} images saved.")
            time.sleep(0.02)
            
        print(f"Final count for {class_name}: {count} images")

if __name__ == "__main__":
    collect_dataset(target_per_class=100)
