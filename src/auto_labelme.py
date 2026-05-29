import os
import json
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO


def generate_perfect_labelme_json(model_path, image_dir, output_dir, conf_threshold=0.35):
    """
    完美适配新版 LabelMe 5.x 的预标注转换脚本
    """
    # 1. 加载模型
    model = YOLO(model_path)
    class_names = model.names
    print(f"成功加载模型，支持类别: {class_names}")

    img_path = Path(image_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG']
    image_files = [f for f in img_path.iterdir() if f.suffix in img_extensions]

    if not image_files:
        print(f"❌ 错误: 在 {image_dir} 目录中没有找到任何照片！")
        return

    print(f"🚀 开始完美转换，预计处理 {len(image_files)} 张图片...")

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w, _ = img.shape

        # 模型推理
        results = model.predict(source=str(img_file), conf=conf_threshold, device=0, verbose=False)
        result = results[0]

        shapes = []
        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                xmin, ymin, xmax, ymax = xyxy

                cls_id = int(box.cls[0].cpu().item())
                label_name = class_names[cls_id]

                # 🎯 核心修复：shapes 内部必须填满新版 LabelMe 要求的线色和填充色控制项
                shape_dict = {
                    "label": label_name,
                    "points": [[xmin, ymin], [xmax, ymax]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "line_color": None,  # 强制对齐新版规范
                    "fill_color": None  # 强制对齐新版规范
                }
                shapes.append(shape_dict)

        # 🎯 核心修复：最外层增加 lineColor、fillColor 以及 imageData 占位符，彻底干掉 'lineColor' 报错！
        labelme_structure = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_file.name,  # 保证单纯的文件名（不要带绝对路径）
            "imageData": None,  # 必须提供，哪怕是 None
            "imageHeight": h,
            "imageWidth": w,
            "lineColor": [0, 255, 0, 128],  # 默认绿色线条 [R, G, B, A]
            "fillColor": [255, 0, 0, 128]  # 默认填充颜色
        }

        # 保存 JSON 到目标文件夹
        json_name = img_file.stem + ".json"
        with open(out_path / json_name, 'w', encoding='utf-8') as f:
            json.dump(labelme_structure, f, ensure_ascii=False, indent=2)

        # 🎯 核心优化：直接把原图片也复制到这个输出文件夹，和 JSON 贴贴，不再分离！
        shutil.copy(str(img_file), str(out_path / img_file.name))

    print(f"\n✅ 完美生成完毕！")
    print(f"👉 【非常重要】接下来请直接用 LabelMe 打开这个文件夹：{output_dir}")


if __name__ == "__main__":
    # ==================== 配置区（请根据实际情况微调） ====================
    MODEL_WEIGHTS = '/home/hnucv/yolo_person_overhead/runs/detect/train6/weights/best.pt'

    # 输入你手拍的干净照片目录
    INPUT_IMAGES = '/home/hnucv/yolo_person_overhead/ourdataset/train/images'

    # 🎯 生成的目标文件夹（图片和完美的 JSON 会成对混在这里面）
    OUTPUT_LABELME = '/home/hnucv/yolo_person_overhead/ourdataset/train/train_labelme_perfect'

    CONFIDENCE = 0.35
    # ====================================================================

    generate_perfect_labelme_json(MODEL_WEIGHTS, INPUT_IMAGES, OUTPUT_LABELME, CONFIDENCE)