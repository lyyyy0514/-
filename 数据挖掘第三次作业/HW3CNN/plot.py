import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载你的完整数据
df = pd.read_csv('D:\DL\HW3CNN\HW3data\single_cell_spatial&celltype.csv')

# 绘图配置
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df, 
    x='x', 
    y='y', 
    hue='cell_type', 
    palette='viridis', # 颜色方案，也可以选 'tab10' 或 'rocket'
    s=1,             # 点的大小，数据量大时建议调小点
    alpha=0.7          # 透明度，防止点太多相互遮挡
)

plt.title('Cell Spatial Map')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 将图例放到外侧
plt.show()