import cv2
import pandas as pd
import os
from tqdm import tqdm
#这个代码主要是对数据进行预处理，首先对高分辨路的图像进行切割成几个低分辨率的图像，然后再根据范围进行划分成训练集、验证集和测试集，最后保存到指定的文件夹中

COOR_CSV_PATH = r"D:\DL\HW3CNN\HW3data\merged_data.csv"
BIG_IMAGE_PATH = r"D:\DL\HW3CNN\HW3data\Xenium_FFPE_Human_Breast_Cancer_Rep1.png"
SAVE_DIR_train = r"D:\DL\HW3CNN\patches_256_train1"
os.makedirs(SAVE_DIR_train, exist_ok=True)#这个是自动创建文件夹的函数，如果文件夹已经存在了，就不会报错
SAVE_DIR_val = r"D:\DL\HW3CNN\patches_256_val1"
os.makedirs(SAVE_DIR_val, exist_ok=True)
SAVE_DIR_test = r"D:\DL\HW3CNN\patches_256_test1"
os.makedirs(SAVE_DIR_test, exist_ok=True)
# 读取坐标
coor_df = pd.read_csv(COOR_CSV_PATH)
print("坐标数据预览：")
print(coor_df.head())

# 直接读取图片
big_img = cv2.imread(BIG_IMAGE_PATH)
if big_img is None:
    raise Exception("图片没找到！检查路径/文件名是否正确")

img_height, img_width = big_img.shape[:2]
print(f" 图片读取成功！尺寸：{img_width} × {img_height}")

# 固定 256×256 裁剪
PATCH_SIZE = 230
HALF = 115

# 开始批量切图
for idx, row in tqdm(coor_df.iterrows(), total=len(coor_df)):
    spot_name = row["Unnamed: 0"]
    cx = round(row["x"])
    cy = round(row["y"])

    # 计算裁剪区域
    x1 = cx - HALF
    y1 = cy - HALF
    x2 = cx + HALF
    y2 = cy + HALF

    # 防止超出图片边界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    # 裁剪
    patch = big_img[y1:y2, x1:x2]
    
    # 尺寸不足256就填充黑色
    h, w = patch.shape[:2]
    if h != PATCH_SIZE or w != PATCH_SIZE:
        patch = cv2.copyMakeBorder(
            patch, 0, PATCH_SIZE-h, 0, PATCH_SIZE-w,
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )

    # 保存
    if cx>17704:
        cv2.imwrite(os.path.join(SAVE_DIR_train, f"{spot_name}.png"), patch)
    elif cx<=17704 and cy<=13015:
        cv2.imwrite(os.path.join(SAVE_DIR_val, f"{spot_name}.png"), patch)
    else:
        cv2.imwrite(os.path.join(SAVE_DIR_test, f"{spot_name}.png"), patch)


print("全部裁剪完成！256×256 patch 已生成")