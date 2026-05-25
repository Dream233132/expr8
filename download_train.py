"""
PlantCLEF 2026 - 训练图片下载脚本

从元数据CSV中的 image_backup_url 列下载训练图片到本地。
为了实际可行性，默认只下载频率最高的 TOP_N_SPECIES 个物种的图片。

目录结构: data/train/{species_id}/{image_name}.jpg
"""

import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 配置
# ============================================================
METADATA_CSV = 'data/PlantCLEF2024_single_plant_training_metadata.csv'
SPECIES_IDS_CSV = 'data/species_ids.csv'
TRAIN_IMG_DIR = 'data/train'

# 下载配置
TOP_N_SPECIES = 200         # 下载频率最高的 N 个物种 (减小可改为50/100)
MAX_IMAGES_PER_SPECIES = 50 # 每个物种最多下载多少张图片
NUM_THREADS = 8             # 并行下载线程数
TIMEOUT = 15                # 单张图片下载超时(秒)

print("=" * 70)
print("PlantCLEF 2026 - Training Image Downloader")
print("=" * 70)
print(f"目标目录: {TRAIN_IMG_DIR}")
print(f"下载物种数: {TOP_N_SPECIES}")
print(f"每物种最多: {MAX_IMAGES_PER_SPECIES} 张")
print(f"并行线程: {NUM_THREADS}")
print(f"预计总下载: ~{TOP_N_SPECIES * MAX_IMAGES_PER_SPECIES} 张图片")

# ============================================================
# 1. 读取元数据
# ============================================================
print("\n[1/3] 读取元数据...")

species_df = pd.read_csv(SPECIES_IDS_CSV)
valid_species = set(species_df['species_id'].tolist())

meta_df = pd.read_csv(
    METADATA_CSV, sep=';',
    usecols=['image_name', 'species_id', 'image_backup_url']
)

# 只保留有效物种
meta_df = meta_df[meta_df['species_id'].isin(valid_species)]
print(f"  有效记录数: {len(meta_df)}")

# 按物种频率排序，取 top N
species_counts = meta_df['species_id'].value_counts()
top_species = species_counts.head(TOP_N_SPECIES).index.tolist()
print(f"  选取物种数: {len(top_species)}")

# 筛选并限制每物种图片数
download_df = meta_df[meta_df['species_id'].isin(top_species)]
download_df = download_df.groupby('species_id').head(MAX_IMAGES_PER_SPECIES)
print(f"  待下载图片数: {len(download_df)}")

# ============================================================
# 2. 创建目录
# ============================================================
print("\n[2/3] 创建目录结构...")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
for sp_id in top_species:
    os.makedirs(os.path.join(TRAIN_IMG_DIR, str(sp_id)), exist_ok=True)

print(f"  已创建 {len(top_species)} 个物种子目录")

# ============================================================
# 3. 下载图片
# ============================================================
print("\n[3/3] 开始下载图片...")

def download_one(row):
    """下载单张图片，返回 (success, path)"""
    sp_id = int(row['species_id'])
    img_name = row['image_name']
    url = row['image_backup_url']
    
    save_path = os.path.join(TRAIN_IMG_DIR, str(sp_id), img_name)
    
    # 跳过已下载的
    if os.path.exists(save_path):
        return (True, save_path, "skip")
    
    try:
        urllib.request.urlretrieve(url, save_path)
        return (True, save_path, "ok")
    except Exception as e:
        return (False, save_path, str(e))


# 构建下载任务列表
tasks = []
for _, row in download_df.iterrows():
    if pd.notna(row['image_backup_url']):
        tasks.append(row)

print(f"  有效下载任务: {len(tasks)}")

# 并行下载
success_count = 0
skip_count = 0
fail_count = 0
start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(download_one, task): task for task in tasks}
    
    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        if result[2] == "skip":
            skip_count += 1
        elif result[0]:
            success_count += 1
        else:
            fail_count += 1
        
        # 每500张打印一次进度
        done = success_count + skip_count + fail_count
        if done % 500 == 0 or done == len(tasks):
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  进度: {done}/{len(tasks)} | "
                  f"成功: {success_count} | 跳过: {skip_count} | "
                  f"失败: {fail_count} | 速度: {speed:.1f} 张/秒")

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"下载完成!")
print(f"  耗时: {elapsed:.0f} 秒 ({elapsed/60:.1f} 分钟)")
print(f"  成功: {success_count}")
print(f"  跳过(已存在): {skip_count}")
print(f"  失败: {fail_count}")
print(f"  总计: {success_count + skip_count} 可用图片")
print(f"\n现在可以运行 python main.py 开始训练了!")
print(f"{'=' * 70}")
