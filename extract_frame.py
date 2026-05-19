#!/usr/bin/env python3
"""
按指定帧率从视频中提取图像帧并保存为图片文件。
使用时间戳均匀采样，保证提取帧率尽可能接近目标值。

依赖库: opencv-python (cv2)
安装: pip install opencv-python
"""

import os
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("错误: 未找到OpenCV库，请执行 'pip install opencv-python' 安装")
    sys.exit(1)


def extract_frames_by_fps(
    video_path: str,
    output_dir: str,
    target_fps: float,
    prefix: str = "frame",
    img_format: str = "jpg",
    quality: int = 95,
    start_time: float = 0.0,
    end_time: float = None,
):
    """
    按目标帧率从视频中提取图像帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        target_fps: 目标提取帧率 (帧/秒)
        prefix: 输出文件名前缀
        img_format: 图片格式 (jpg, png, bmp等)
        quality: JPEG压缩质量 (1-100)，仅对jpg有效
        start_time: 起始提取时间(秒)，默认从0开始
        end_time: 结束提取时间(秒)，None表示到视频结尾
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    # 获取视频属性
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    if video_fps <= 0:
        cap.release()
        raise ValueError("无法获取视频帧率")

    print(f"视频信息: {video_path}")
    print(f"  帧率: {video_fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f} 秒")
    print(f"目标帧率: {target_fps} fps")
    print(f"输出目录: {output_dir}")

    # 处理目标帧率大于视频帧率的情况：提取所有帧
    if target_fps >= video_fps:
        print("目标帧率 >= 视频原始帧率，将提取所有帧")
        extract_all = True
        interval_sec = 0  # 未使用
    else:
        extract_all = False
        interval_sec = 1.0 / target_fps

    # 设置编码参数（JPEG质量）
    encode_param = None
    if img_format.lower() in ["jpg", "jpeg"]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # 定位到起始时间（如果指定）
    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        print(f"从 {start_time:.2f} 秒处开始提取")

    # 初始化变量
    extracted_count = 0
    frame_idx = 0  # 当前帧序号（仅用于显示）
    next_extract_time = None  # 下一次提取的理想时间戳（秒）
    first_frame_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的时间戳（秒）
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cur_time = timestamp_ms / 1000.0

        # 检查是否超出结束时间
        if end_time is not None and cur_time > end_time:
            break

        frame_idx += 1

        # 跳过起始时间之前的帧
        if cur_time < start_time:
            continue

        # 判断是否需要保存当前帧
        save_this_frame = False

        if extract_all:
            save_this_frame = True
        else:
            # 第一帧直接保存作为起始点
            if not first_frame_saved:
                save_this_frame = True
                first_frame_saved = True
                # 设置下一次提取的理想时间
                next_extract_time = cur_time + interval_sec
            else:
                # 当前帧时间戳达到了下一次提取的理想时间
                if cur_time >= next_extract_time:
                    save_this_frame = True
                    # 更新下一次提取时间（允许多周期，但不重复保存同一帧）
                    # 由于帧时间戳是离散的，这里简单加上间隔
                    next_extract_time += interval_sec

        if save_this_frame:
            extracted_count += 1
            # 生成文件名: prefix_序号.扩展名
            filename = f"{prefix}_{extracted_count:06d}.{img_format}"
            filepath = output_path / filename

            # 保存图片
            if encode_param:
                success = cv2.imwrite(str(filepath), frame, encode_param)
            else:
                success = cv2.imwrite(str(filepath), frame)

            if success:
                print(f"[{extracted_count}] 保存: {filepath} (时间: {cur_time:.3f}s)")
            else:
                print(f"警告: 保存失败 {filepath}")

    cap.release()
    print(f"\\n完成! 共提取 {extracted_count} 帧，保存至 {output_dir}")


if __name__ == "__main__":
    # ==================== 用户配置区域 ====================
    # 请在这里设置视频文件路径（必填）
    file_path = r"/home/hnucv/yolo_person_overhead/dataset/test/video/3.avi"   # 修改为您的视频文件路径

    # 可选参数（根据需要修改）
    output_dir = r"/home/hnucv/yolo_person_overhead/dataset/test/new_images"   # 输出目录
    target_fps = 10                 # 目标提取帧率 (帧/秒)
    prefix = "frame1478"                  # 输出文件名前缀
    img_format = "jpg"                # 图片格式: jpg, png, bmp
    quality = 100                      # JPEG质量 (1-100)
    start_time = 0.0                  # 起始时间(秒)
    end_time = None                   # 结束时间(秒)，None表示到视频结尾
    # =====================================================

    # 检查视频文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 视频文件不存在: {file_path}")
        print("请修改 file_path 变量为正确的视频路径")
        sys.exit(1)

    if target_fps <= 0:
        print("错误: 目标帧率必须大于0")
        sys.exit(1)

    try:
        extract_frames_by_fps(
            video_path=file_path,
            output_dir=output_dir,
            target_fps=target_fps,
            prefix=prefix,
            img_format=img_format,
            quality=quality,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)