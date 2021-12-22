# encoding=utf-8
from PIL import Image, ImageChops
import cv2
import numpy as np
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim

original_path = "D:\Program\Videos\Tools\PotPlayer\Capture\偏色.flv_20211222_185644.241.png"
rendered = "D:\Program\Videos\Tools\PotPlayer\Capture\偏色_48fps_[S-1.0]_[official_4.0]_753360.mp4_20211222_190535.659.png"
rendered = "D:\Program\Videos\Tools\PotPlayer\Capture\偏色_48fps_[S-1.0]_[official_4.0]_389760.mp4_20211222_192330.383.png"
def compare():
    img1 = Image.open(original_path)
    img2 = Image.open(rendered)
    diff = ImageChops.difference(img1, img2)
    diff.save('diff2.png')

def imread(path):
    return  cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)[:, :, ::-1].copy()

def meann():
    img1 = imread('./diff-8bit,关快拆硬解.png')
    img2 = imread('./diff-10bit,关快拆硬解.png')
    print(f"slow rgb24: mean: {img1.mean()} var: {np.var(img1)}")
    print(f"slow rgb48: mean: {img2.mean()} var: {np.var(img2)}")
    img1 = imread(original_path)
    img2 = imread(rendered)
    print(f"ssim: {compare_ssim(img1, img2, multichannel=True)}")

# compare()
meann()