
import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

def save_image_pil(image, filename, DPI=600):
    if image.shape[2] == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)
    pil_img.save(filename, format='PNG', dpi=(DPI, DPI))