import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# ===================== 1. 核心参数配置（正六边形 + 单位换算）=====================
# 单位换算：55μm=259px → 100μm=470.91px（正六边形中心到顶点的距离=外接圆半径R）
HEX_RADIUS = 471  # ≈470.91 像素（正六边形关键参数）
SQRT3 = np.sqrt(3)  # 根号3，正六边形几何计算常量

# 文件路径（完全沿用你的路径）
SPOT_CSV_PATH = r"D:\DL\HW3CNN\HW3data\coor.csv"
CELL_CSV_PATH = r"D:\DL\HW3CNN\HW3data\single_cell_spatial&celltype.csv"

# ===================== 2. 加载数据 =====================
# 读取Spot中心坐标（正六边形中心）
spot_df = pd.read_csv(SPOT_CSV_PATH, index_col=0)
spot_centers = spot_df[['x', 'y']].values
print(f"✅ 成功加载 {len(spot_df)} 个Spot（正六边形）中心坐标")

# 读取细胞坐标+类型数据
cell_df = pd.read_csv(CELL_CSV_PATH)
cell_coords = cell_df[['x', 'y']].values
print(f"✅ 成功加载 {len(cell_df)} 个细胞数据")

# ===================== 3. 极速匹配：KDTree 找最近的Spot =====================
# 构建空间索引，为每个细胞找到最近的正六边形中心
tree = KDTree(spot_centers)
nearest_spot_ids = tree.query(cell_coords, k=1)[1]
nearest_spots = spot_centers[nearest_spot_ids]

# ===================== 4. 核心修改：正六边形区域判断（向量化计算，极速）=====================
# 提取坐标差值
cx, cy = nearest_spots[:, 0], nearest_spots[:, 1]
x, y = cell_coords[:, 0], cell_coords[:, 1]

dx = np.abs(x - cx)
dy = np.abs(y - cy)

# 标准水平正六边形（顶点上下左右）的点包含判断公式
in_hexagon = (
    # 条件1：垂直方向不超过正六边形高度
    (dy <= SQRT3 / 2 * HEX_RADIUS) &
    # 条件2：斜向边界约束（核心判断）
    (2 * dx + dy <= SQRT3 * HEX_RADIUS)
)

# ===================== 5. 标注细胞所属Spot =====================
cell_df['belong_spot'] = np.where(
    in_hexagon,
    spot_df.index[nearest_spot_ids],
    "未匹配"
)

# ===================== 6. 统计每个Spot的细胞数量 =====================
matched_cells = cell_df[cell_df['belong_spot'] != "未匹配"]
spot_cell_count = matched_cells['belong_spot'].value_counts().sort_index()

count_result = pd.DataFrame({
    'spot_id': spot_cell_count.index,
    'cell_count': spot_cell_count.values
})

# ===================== 7. 输出结果 + 保存文件 =====================
print("\n" + "="*60)
print("📊 每个正六边形Spot的细胞数量统计：")
print(count_result.to_string(index=False))
print("="*60)

# 保存文件（沿用你的命名）
cell_df.to_csv(r"D:\DL\HW3CNN\HW3data\细胞归属Spot结果.csv", index=False, encoding='utf-8-sig')
count_result.to_csv(r"D:\DL\HW3CNN\HW3data\Spot细胞数量统计.csv", index=False, encoding='utf-8-sig')

print("\n🎉 处理完成！")
print(f"📌 总匹配成功细胞数：{len(matched_cells)} 个")
print(f"📌 未匹配细胞数：{len(cell_df[cell_df['belong_spot'] == '未匹配'])} 个")