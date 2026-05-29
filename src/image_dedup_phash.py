import os
import cv2
from pathlib import Path
# 🎯 引入进度条库
from tqdm import tqdm

def calculate_phash(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32))
    img = img.astype(float)
    dct = cv2.dct(img)
    dct_low = dct[0:8, 0:8]
    avg = dct_low.mean()
    phash_str = ''.join(['1' if x > avg else '0' for x in dct_low.flatten()])
    return phash_str

LABELED_DIR = '/home/hnucv/yolo_person_overhead/1000/image2'
RAW_5000_DIR = '/home/hnucv/yolo_person_overhead/ourdataset/train/images'

img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG'}

# 1. 扫描 1000 张已标集（加进度条）
labeled_files = [f for f in Path(LABELED_DIR).iterdir() if f.suffix in img_exts]
print(f"🚀 开始提取 {len(labeled_files)} 张已标集的特征指纹...")
labeled_hashes = set()

for f in tqdm(labeled_files, desc="提取精标集指纹"):
    h_str = calculate_phash(f)
    if h_str:
        labeled_hashes.add(h_str)

# 2. 扫描并清洗 5000 张原始图（加进度条）
raw_files = [f for f in Path(RAW_5000_DIR).iterdir() if f.suffix in img_exts]
print(f"\n📊 提取完毕。开始对比并清洗 {len(raw_files)} 张原始图...")

delete_count = 0
for f_raw in tqdm(raw_files, desc="比对并剔除重复"):
    h_raw = calculate_phash(f_raw)
    if h_raw in labeled_hashes:
        f_raw.unlink()  # 物理删除
        delete_count += 1

print(f"\n✅ 视觉去重完毕！共删除了 {delete_count} 张重复图片！")