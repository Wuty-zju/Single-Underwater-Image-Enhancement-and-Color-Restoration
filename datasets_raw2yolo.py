import os
import shutil
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm
import urllib.request
import zipfile

# 全局参数设置
random_seed = 42
train_ratio = 0.7  # 训练集比例
val_ratio = 0.2    # 验证集比例
test_ratio = 0.1   # 测试集比例
base_url = "http://www.caddian.eu//assets/caddy-gestures-TMP/CADDY_gestures_complete_v2_release.zip"
expected_file_count = 32860  # 目标文件数量

# 定义目录结构
project_dir = os.path.normpath('')
datasets_dir = os.path.join(project_dir, 'datasets')
base_dir = os.path.join(datasets_dir, 'CADDY_gestures_complete')
raw_dir = os.path.join(base_dir, 'raw')
yolo_dir = os.path.join(base_dir, 'yolo')
zip_file_path = os.path.join(base_dir, 'CADDY_gestures_complete_v2_release.zip')
tn_input_csv = os.path.join(raw_dir, 'CADDY_gestures_all_true_negatives_release_v2.csv')
tp_input_csv = os.path.join(raw_dir, 'CADDY_gestures_all_true_positives_release_v2.csv')
output_images_dir = os.path.join(yolo_dir, 'datasets', 'images')
output_labels_dir = os.path.join(yolo_dir, 'datasets', 'labels')
train_file = os.path.join(yolo_dir, 'train.txt')
val_file = os.path.join(yolo_dir, 'val.txt')
test_file = os.path.join(yolo_dir, 'test.txt')

# 确保目录存在
os.makedirs(base_dir, exist_ok=True)
os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

def download_and_extract(url, zip_path, extract_to, use_proxy=False):
    """
    下载并解压数据集，显示下载进度和传输速度。
    
    参数:
        url (str): 下载链接。
        zip_path (str): 下载的文件保存路径。
        extract_to (str): 解压目录。
        use_proxy (bool): 是否使用代理下载，默认为False。
    """
    # 设置代理
    if use_proxy:
        proxy = urllib.request.ProxyHandler({'http': 'http://192.168.31.229:7222'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
        print("Using HTTP proxy for download.")

    # 下载文件
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB per block

        with open(zip_path, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading", ncols=80
        ) as progress_bar:
            for data in iter(lambda: response.read(block_size), b''):
                file.write(data)
                progress_bar.update(len(data))
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

    # 解压文件并校验文件数量
    def count_files_in_directory(directory):
        return sum([len(files) for _, _, files in os.walk(directory)])

    if os.path.exists(extract_to) and count_files_in_directory(extract_to) == expected_file_count:
        print(f"Extraction directory '{extract_to}' already contains {expected_file_count} files. Skipping extraction.")
    else:
        if os.path.exists(extract_to):
            print(f"File count mismatch in '{extract_to}'. Re-extracting...")
            shutil.rmtree(extract_to)
        os.makedirs(extract_to, exist_ok=True)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取最外层文件夹名称
            top_level_dir = zip_ref.namelist()[0].split('/')[0]
            for member in tqdm(zip_ref.infolist(), desc="Extracting", unit="files", ncols=80):
                # 修改解压路径，忽略最外层文件夹
                member_path = member.filename
                if member_path.startswith(top_level_dir):
                    member_path = member_path[len(top_level_dir) + 1:]
                target_path = os.path.join(extract_to, member_path)
                if member_path:
                    # 如果member_path不是空字符串，则解压
                    zip_ref.extract(member, extract_to)
                    # 移动文件到正确的位置
                    os.rename(os.path.join(extract_to, member.filename), target_path)
        print("Extraction complete.")

    # 校验解压后的文件数量
    extracted_file_count = count_files_in_directory(extract_to)
    if extracted_file_count == expected_file_count:
        print(f"Extraction complete with {extracted_file_count} files.")
    else:
        print(f"Warning: Expected {expected_file_count} files, but found {extracted_file_count} after extraction.")
# 调用下载和解压函数
download_and_extract(base_url, zip_file_path, raw_dir, use_proxy=True)

# 继续处理数据集
def process_dataset(df, prefix, is_tp=True):
    """
    处理数据集，将图片和标签转换为YOLO格式并保存
    """
    converted_images = 0
    converted_labels = 0
    discarded_labels_count = 0
    not_found_images = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {prefix.upper()} Dataset"):
        # 图像路径
        image_path_left = os.path.join(raw_dir, row['stereo left'].strip().lstrip('/'))
        image_path_right = os.path.join(raw_dir, row['stereo right'].strip().lstrip('/'))

        # 检查文件是否存在
        if not os.path.exists(image_path_left) or not os.path.exists(image_path_right):
            not_found_images.extend([image_path_left, image_path_right])
            continue

        # 复制图像并重命名
        target_path_left = shutil.copy(image_path_left, os.path.join(output_images_dir, f"{prefix}_{os.path.basename(image_path_left)}"))
        target_path_right = shutil.copy(image_path_right, os.path.join(output_images_dir, f"{prefix}_{os.path.basename(image_path_right)}"))

        # 获取图像尺寸
        try:
            with Image.open(target_path_left) as img:
                image_width_left, image_height_left = img.size
            with Image.open(target_path_right) as img:
                image_width_right, image_height_right = img.size
        except FileNotFoundError:
            not_found_images.extend([target_path_left, target_path_right])
            continue

        # 提取ROI信息并转换为YOLO格式
        def process_roi(roi_str, width, height):
            yolo_lines = []
            if isinstance(roi_str, str) and roi_str:
                for roi_part in roi_str.strip('[]').split(';'):
                    x, y, w, h = map(int, roi_part.split(','))
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    width_ratio = w / width
                    height_ratio = h / height
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and width_ratio <= 1 and height_ratio <= 1:
                        yolo_lines.append(f"{row['label id']} {x_center} {y_center} {width_ratio} {height_ratio}")
                    else:
                        nonlocal discarded_labels_count
                        discarded_labels_count += 1
            return yolo_lines

        # 判断是TP还是TN，选择正确的ROI列
        label_lines_left = process_roi(row['roi left'], image_width_left, image_height_left) if is_tp else []
        label_lines_right = process_roi(row['roi right'], image_width_right, image_height_right) if is_tp else []

        # 保存标签文件
        for label_lines, target_path in [(label_lines_left, target_path_left), (label_lines_right, target_path_right)]:
            label_file = os.path.splitext(os.path.basename(target_path))[0] + '.txt'
            label_file_path = os.path.join(output_labels_dir, label_file)
            with open(label_file_path, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')

        converted_images += 2
        converted_labels += len(label_lines_left) + len(label_lines_right)

    print(f"Processed {prefix.upper()} Dataset: {converted_images} images, {converted_labels} labels, {discarded_labels_count} discarded, {len(not_found_images)} not found")

# 加载数据集并处理
tp_df = pd.read_csv(tp_input_csv)
tn_df = pd.read_csv(tn_input_csv)
process_dataset(tp_df, 'tp', is_tp=True)
process_dataset(tn_df, 'tn', is_tp=False)

def split_dataset(images_dir, train_file, val_file, test_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    随机划分数据集为训练集、验证集和测试集，并生成相应的txt文件
    """
    all_images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
    paired_images = list(zip(all_images[::2], all_images[1::2]))  # 保证左右图像成对
    random.seed(random_seed)
    random.shuffle(paired_images)

    train_split = int(len(paired_images) * train_ratio)
    val_split = int(len(paired_images) * (train_ratio + val_ratio))

    train_images = paired_images[:train_split]
    val_images = paired_images[train_split:val_split]
    test_images = paired_images[val_split:]

    def save_split(images, filepath):
        with open(filepath, 'w') as f:
            for left, right in images:
                f.write(f'{os.path.relpath(left, yolo_dir).replace("\\", "/")}\n')
                f.write(f'{os.path.relpath(right, yolo_dir).replace("\\", "/")}\n')

    save_split(train_images, train_file)
    save_split(val_images, val_file)
    save_split(test_images, test_file)

split_dataset(output_images_dir, train_file, val_file, test_file, train_ratio, val_ratio, test_ratio)

def generate_yaml(labels_df, output_dir, train_file, val_file, test_file):
    """
    生成YOLO格式的数据集配置文件
    """
    nc = labels_df['label id'].max() + 1
    names = '\n'.join([f"  {row['label id']}: {row['label name']}" for _, row in labels_df.drop_duplicates(subset='label id').sort_values('label id').iterrows()])
    yaml_content = f"""\
# CADDY Underwater Gestures Dataset
path: {os.path.relpath(output_dir, project_dir).replace("\\", "/")}
train: {os.path.relpath(train_file, project_dir).replace("\\", "/")}
val: {os.path.relpath(val_file, project_dir).replace("\\", "/")}
test: {os.path.relpath(test_file, project_dir).replace("\\", "/")}

nc: {nc}
names:
{names}
"""
    yaml_file = os.path.join(output_dir, 'CADDY_gestures_complete.yaml')
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)

generate_yaml(tp_df[['label name', 'label id']], yolo_dir, train_file, val_file, test_file)

print("数据集处理完成，配置文件生成完毕。")