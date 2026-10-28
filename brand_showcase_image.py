"""Create a branded variant with Apple-style sleek frosted badge overlay."""
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "/Users/erbiumtech/Downloads/Florix_AI_Showcase.png"
DST = "/Users/erbiumtech/Downloads/Florix_AI_Showcase_Branded.png"

im = Image.open(SRC).convert("RGBA")
W, H = im.size

overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
def get_f(sz, bold=False):
    idx = 1 if bold else 0
    try:
        return ImageFont.truetype(FONT_PATH, sz, index=idx)
    except:
        return ImageFont.load_default()

# 1. Subtle top-left badge pill
pad_w, pad_h = 480, 50
draw.rounded_rectangle([60, 60, 60 + pad_w, 60 + pad_h], radius=25, fill=(10, 15, 30, 200), outline=(52, 211, 153, 120), width=1)
draw.ellipse([80, 80, 80 + 10, 80 + 10], fill=(52, 211, 153, 255))
draw.text((102, 75), "DEEP LEARNING SEMESTER PROJECT • PROF. DR MUNIR AHMAD", font=get_f(12, bold=True), fill=(255, 255, 255, 240))

# 2. Sleek bottom floating glassmorphic strip
bw, bh = W - 120, 130
bx, by = 60, H - 190
draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=22, fill=(8, 12, 28, 215), outline=(255, 255, 255, 35), width=1)

# Left text
draw.text((bx + 35, by + 26), "🌸 FLORIX AI — Botanical Flower Classification", font=get_f(28, bold=True), fill=(255, 255, 255, 255))
draw.text((bx + 35, by + 74), "15,276 Images  •  MobileNetV2: 85.11% Acc  •  Custom CNN: 75.29%  •  0.0% Non-Flower Guarantee", font=get_f(15), fill=(148, 163, 184, 255))

# Right student badge
draw.text((bx + bw - 380, by + 34), "Ahmad Ishtiaq", font=get_f(20, bold=True), fill=(52, 211, 153, 255))
draw.text((bx + bw - 380, by + 68), "SU92-BSCSM-F23-318  |  BSCS-7A", font=get_f(13), fill=(203, 213, 225, 240))

final_im = Image.alpha_composite(im, overlay)
final_im.convert("RGB").save(DST, "PNG", quality=95)
print("✅ Saved branded variant to:", DST)
