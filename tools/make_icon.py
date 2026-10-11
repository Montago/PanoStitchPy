"""Generate the app icon (app/icons/app.png + app.ico): a blue rounded square with a horizon and '360°'."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "app" / "icons"
S = 1024  # draw large, downsample for smooth edges

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 40, S - 40, S - 40], radius=200, fill=(47, 127, 224, 255))
# sky gradient-ish band and a curved horizon (like an equirectangular panorama)
d.rounded_rectangle([40, 40, S - 40, 560], radius=200, fill=(92, 162, 240, 255))
d.rectangle([40, 360, S - 40, 560], fill=(92, 162, 240, 255))
d.ellipse([-300, 520, S + 300, 1500], fill=(46, 140, 90, 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=200, fill=255)
img.putalpha(Image.composite(img.getchannel("A"), mask, mask))

try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 330)
except OSError:
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 330)
text = "360°"
box = d.textbbox((0, 0), text, font=font)
x = (S - (box[2] - box[0])) / 2 - box[0]
y = 250 - box[1]
d.text((x + 10, y + 12), text, font=font, fill=(0, 0, 0, 90))
d.text((x, y), text, font=font, fill=(255, 255, 255, 255))

OUT.mkdir(parents=True, exist_ok=True)
icon = img.resize((256, 256), Image.LANCZOS)
icon.save(OUT / "app.png")
icon.save(OUT / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", OUT / "app.png", OUT / "app.ico")
