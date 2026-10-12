import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 基础参数 =====================
circle_diameter = 259
square_side = 256
r = circle_diameter / 2  # 圆半径 129.5
s_half = square_side / 2 # 正方形半边长 128.0
area_square = square_side ** 2  # 正方形总面积

# ===================== 方法1：精准解析法（积分/弓形，100%正确） =====================
x0 = math.sqrt(r**2 - s_half** 2)  # 分界点x坐标

# 第一象限交集面积（严格积分/几何推导）
part1 = x0 * s_half  # 矩形区域面积
part2 = 0.5 * ( s_half * math.sqrt(r**2 - s_half **2) + r**2 * math.asin(s_half / r) 
                - x0 * s_half - r** 2 * math.asin(x0 / r) )
quadrant_area = part1 + part2
area_intersection = quadrant_area * 4  # 四象限总面积

# 覆盖率：交集面积 ÷ 正方形面积
coverage_analytic = (area_intersection / area_square) * 100

# ===================== 方法2：蒙特卡洛模拟（撒点验证，权威正确） =====================
def monte_carlo(n=5_000_000):
    count = 0
    for _ in range(n):
        x = random.uniform(-s_half, s_half)
        y = random.uniform(-s_half, s_half)
        if x**2 + y** 2 <= r**2:
            count += 1
    return (count / n) * 100

coverage_monte = monte_carlo()

# ===================== 输出结果 =====================
print("="*60)
print(f"圆直径：{circle_diameter} | 半径：{r}")
print(f"正方形边长：{square_side} | 总面积：{area_square:.2f}")
print(f"✅ 精准解析法 交集面积：{area_intersection:.2f} | 覆盖率：{coverage_analytic:.2f}%")
print(f"✅ 蒙特卡洛模拟 覆盖率：{coverage_monte:.2f}%")
print("="*60)

# ===================== 绘图 =====================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
circle = Circle((0,0), r, color='red', alpha=0.5, label=f'圆(直径{circle_diameter})')
square = Rectangle((-s_half,-s_half), square_side, square_side, color='blue', alpha=0.5, label=f'正方形(边长{square_side})')
ax.add_patch(circle)
ax.add_patch(square)
ax.set_xlim(-150,150)
ax.set_ylim(-150,150)
ax.set_title(f'相交覆盖率：{coverage_analytic:.2f}%', fontsize=14)
ax.legend()
plt.grid(alpha=0.3)
plt.show()


import math
import matplotlib.pyplot as plt
import numpy as np

# ===================== 全局设置：中文字体 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ===================== 基础参数：固定圆 =====================
circle_diameter = 259
R = circle_diameter / 2  # 圆半径 129.5
circle_area = math.pi * R ** 2

# ===================== 核心：覆盖率计算函数（精准解析法） =====================
def calculate_coverage(square_side):
    """
    计算正方形被圆覆盖的覆盖率（交集/正方形面积）
    :param square_side: 正方形边长
    :return: 覆盖率（0-1之间）
    """
    s = square_side / 2  # 正方形半边长
    area_square = square_side ** 2

    # 情况1：正方形完全在圆内（边长 ≤ 临界值）
    if square_side <= R * math.sqrt(2):
        return 1.0  # 覆盖率100%

    # 情况2：正方形顶点在圆外，边在圆内（相交）
    elif s < R:
        x0 = math.sqrt(R ** 2 - s ** 2)  # 临界x坐标
        # 第一象限积分面积（精准推导）
        part1 = x0 * s  # 矩形部分
        # 圆弧积分部分（圆缺/弓形，本质是积分）
        angle1 = math.asin(s / R)
        angle2 = math.asin(x0 / R)
        part2 = 0.5 * (s * math.sqrt(R ** 2 - s ** 2) + R ** 2 * angle1 - x0 * s - R ** 2 * angle2)
        quadrant_area = part1 + part2  # 第一象限交集面积
        inter_area = quadrant_area * 4  # 四象限总面积
        return inter_area / area_square

    # 情况3：正方形完全包裹圆（边长 > 2R，极少出现）
    else:
        return circle_area / area_square

# ===================== 1. 绘制：覆盖率随边长变化曲线 =====================
# 生成边长范围（100 到 300，覆盖所有关键情况）
side_list = np.linspace(100, 300, 300)
coverage_list = [calculate_coverage(side) for side in side_list]

# 临界边长：正方形顶点刚好贴圆（覆盖率100%的最大值边界）
critical_side = R * math.sqrt(2)
max_coverage = 1.0

# 创建画布
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 绘制覆盖率曲线
ax1.plot(side_list, coverage_list, color='#2E86AB', linewidth=2.5, label='覆盖率曲线')
# 标注临界边长（最大值边界）
ax1.axvline(critical_side, color='#A23B72', linestyle='--', linewidth=2, label=f'临界边长 ≈ {critical_side:.2f}')
# 标注最大覆盖率
ax1.axhline(max_coverage, color='#F18F01', linestyle='--', linewidth=2, label='最大覆盖率 = 100%')

# 图表美化
ax1.set_xlabel('正方形边长', fontsize=12)
ax1.set_ylabel('覆盖率（交集面积/正方形面积）', fontsize=12)
ax1.set_title('正方形被圆覆盖的覆盖率随边长变化曲线', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 1.05)  # 限制y轴范围，更直观

# ===================== 2. 绘制：三种典型边长的几何对比图 =====================
# 选取三种典型边长
sides = [
    150,  # 情况1：完全在圆内（覆盖率100%）
    critical_side,  # 情况2：临界贴圆（覆盖率100%，边界）
    256  # 情况3：相交露角（覆盖率<100%，你之前的案例）
]
labels = ['完全在圆内（边长150）', '临界贴圆（边长≈183.14）', '相交露角（边长256）']
colors = ['#4CAF50', '#FF9800', '#F44336']

# 创建子图
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (side, label, color) in enumerate(zip(sides, labels, colors)):
    ax = axes[i]
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=12, fontweight='bold')

    # 绘制圆
    circle = plt.Circle((0, 0), R, color='red', alpha=0.5, label='圆')
    ax.add_patch(circle)

    # 绘制正方形
    s = side / 2
    square = plt.Rectangle((-s, -s), side, side, color=color, alpha=0.3, label='正方形')
    ax.add_patch(square)

    # 坐标范围
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    # 辅助线
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    # 标注覆盖率
    cov = calculate_coverage(side) * 100
    ax.text(0, 130, f'覆盖率：{cov:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')

# 整体布局调整
plt.tight_layout()
# 显示所有图形
plt.show()

# ===================== 输出关键结论 =====================
print("="*60)
print(f"圆半径：{R:.2f}，圆总面积：{circle_area:.2f}")
print(f"✅ 最大覆盖率：100%（达到上限，无法更高）")
print(f"✅ 达到最大覆盖率的临界正方形边长：{critical_side:.2f}")
print(f"✅ 边长 ≤ {critical_side:.2f} 时，正方形完全在圆内，覆盖率恒为100%")
print(f"✅ 边长 > {critical_side:.2f} 后，覆盖率随边长增大而持续下降")
print("="*60)