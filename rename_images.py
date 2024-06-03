import os


def rename_images(directory, prefix, start_index=1):
    """
    批量更改图片文件名称
    :param directory: 图片文件所在目录
    :param prefix: 新文件名前缀
    :param start_index: 起始索引
    """
    # 获取目录中的所有文件
    files = os.listdir(directory)

    # 过滤出图片文件（假设图片文件扩展名为.jpg, .png等）
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    # 按照文件名排序
    image_files.sort()

    # 批量重命名图片文件
    for i, filename in enumerate(image_files, start=start_index):
        # 获取文件扩展名
        file_ext = os.path.splitext(filename)[1]

        # 构建新的文件名
        new_name = f"{prefix}_{i}{file_ext}"

        # 构建完整的文件路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")


# 示例用法
directory = 'datasets/background'
prefix = 'image'
start_index = 1

rename_images(directory, prefix, start_index)