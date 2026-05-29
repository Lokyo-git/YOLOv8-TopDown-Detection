import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm  # 🎯 加上进度条


def convert_labelme_to_yolo(json_dir, output_images_dir, output_labels_dir):

    json_path = Path(json_dir)
    out_img_path = Path(output_images_dir)
    out_txt_path = Path(output_labels_dir)

    out_img_path.mkdir(parents=True, exist_ok=True)
    out_txt_path.mkdir(parents=True, exist_ok=True)

    # 自动生成类别映射表
    class_to_id = {'person': 0}

    # 兼容大小写后缀
    json_files = list(json_path.glob("*.json")) + list(json_path.glob("*.JSON"))
    print(f"📦 正在解析 LabelMe 数据集，找到 {len(json_files)} 个标注文件...")

    success_count = 0

    for j_file in tqdm(json_files, desc="YOLO格式转化中"):
        with open(j_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_name = data.get("imagePath")
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")

        img_src_path = json_path / img_name
        if not img_w or not img_h:
            img = cv2.imread(str(img_src_path))
            if img is None:
                print(f"⚠️ 找不到对应的原图 {img_name}，跳过该文件")
                continue
            img_h, img_w, _ = img.shape

        yolo_lines = []

        # 2. 遍历 LabelMe 里的每一个标注组件
        for shape in data.get("shapes", []):
            label = shape.get("label")
            shape_type = shape.get("shape_type")
            points = shape.get("points")

            if label not in class_to_id or not points or len(points) < 2:
                continue

            cls_id = class_to_id[label]

            # 🎯 【核心兼容升级】：如果是普通的标准矩形
            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points[0], points[1]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)

            # 🎯 【核心兼容升级】：如果是画歪了的多边形 (Polygon)
            elif shape_type == "polygon":
                # 提取多边形所有顶点的 x 坐标和 y 坐标
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                # 自动计算能够框住这个多边形的最小外接矩形边界
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)

            else:
                # 其他不认识的标注类型（如线条、点）跳过
                continue

            # 计算 YOLO 需要的中心点、宽高，并归一化
            box_w = xmax - xmin
            box_h = ymax - ymin
            x_center = xmin + box_w / 2.0
            y_center = ymin + box_h / 2.0

            x_center_norm = x_center / img_w
            y_center_norm = y_center / img_h
            box_w_norm = box_w / img_w
            box_h_norm = box_h / img_h

            line_str = f"{cls_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}"
            yolo_lines.append(line_str)

        # 🎯 强行加上前缀 't_' 生成新的 TXT 文件名
        txt_name = "v_" + j_file.stem + ".txt"
        with open(out_txt_path / txt_name, 'w', encoding='utf-8') as f_txt:
            f_txt.write("\n".join(yolo_lines))

        # 🎯 强行加上前缀 't_' 生成新的图片文件名
        new_img_name = "v_" + img_name

        # 4. 把原图片复制到最终的 YOLO images 目录下
        img_mat = cv2.imread(str(img_src_path))
        if img_mat is not None:
            cv2.imwrite(str(out_img_path / new_img_name), img_mat)
            success_count += 1

    print(f"\n🎉 转换大获成功！")
    print(f"✅ 成功将 {success_count} 张手拍图（完美兼容多边形）处理为带 'v_' 前缀的标准 YOLO 格式！")
    print(f"📂 最终图片: {output_images_dir}")
    print(f"📂 最终 TXT 标签: {output_labels_dir}")


if __name__ == "__main__":
    # ==================== 🛠️ 最终格式整合配置区 ====================
    # 保持你目前的路径设定
    LABELME_WORK_DIR = '/home/hnucv/yolo_person_overhead/1000/valid_labelme_perfect'

    YOLO_FINAL_IMAGES = '/home/hnucv/yolo_person_overhead/1000/images'
    YOLO_FINAL_LABELS = '/home/hnucv/yolo_person_overhead/1000/labels'
    # =============================================================

    convert_labelme_to_yolo(
        json_dir=LABELME_WORK_DIR,
        output_images_dir=YOLO_FINAL_IMAGES,
        output_labels_dir=YOLO_FINAL_LABELS
    )