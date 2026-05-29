import os
from pathlib import Path
from tqdm import tqdm  # 🎯 导入进度条库


def delete_files_with_prefix(target_dir, prefix="frame5"):
    """
    删除指定文件夹下所有以特定前缀（如 frame5）开头的文件，并显示进度条
    """
    dir_path = Path(target_dir)

    if not dir_path.exists():
        print(f"❌ 错误：找不到路径 {target_dir}，请检查是否拼写错误！")
        return

    # 1. 挖出文件夹下所有文件
    all_files = [f for f in dir_path.iterdir() if f.is_file()]

    # 2. 过滤出真正符合“以 frame5 开头”的目标文件
    target_files = [f for f in all_files if f.name.startswith(prefix)]

    total_targets = len(target_files)
    if total_targets == 0:
        print(f"🔍 扫描完毕：在当前目录下没有找到任何以 '{prefix}' 开头的文件，无需删除。")
        return

    print(f"🚀 找到 {total_targets} 个以 '{prefix}' 开头的文件。开始执行物理删除...")

    delete_count = 0
    # 3. 带进度条的删除循环
    for file_to_del in tqdm(target_files, desc="正在清理指定前缀文件"):
        try:
            file_to_del.unlink()  # ❌ 物理删除文件
            delete_count += 1
        except Exception as e:
            print(f"\n⚠️ 删除文件 {file_to_del.name} 失败，原因: {e}")

    print(f"\n✅ 清理完毕！成功删除了 {delete_count} 个以 '{prefix}' 开头的文件，其他文件已全部安全保留。")


if __name__ == "__main__":
    # ==================== 🛠️ 路径配置区 ====================
    # 🎯 把这里改成你想清理的那个数据集文件夹路径（比如你们的 train/images 或者 labels）
    CLEAN_DIR = '/home/hnucv/yolo_person_overhead/ourdataset/train/images_old'
    # ====================================================

    # 执行删除
    delete_files_with_prefix(target_dir=CLEAN_DIR, prefix="frame5")