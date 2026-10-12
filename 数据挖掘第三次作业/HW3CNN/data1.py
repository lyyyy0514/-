import pandas as pd
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np 
from scipy.spatial import KDTree
# 接下里的任务是预测每个spot内的细胞数量
#主要的思路是把signle_cell_spatial的数据归类到spot内，如果在范围内就记一个数，这样可以得到不同类型细胞的数量，建立一个监督模型
# 第一步，看有几个种类，种类数就是文件中所有种类的unqiue，这个不需要找缺失值，因为如果有缺失值也不好补
# 首先读取两个csv文件，然后进行合并，最后查看缺失值的情况以及缺失值的比例，最后查看前20行数据的情况，以便后续处理
df1 = pd.read_csv(r"D:\DL\HW3CNN\HW3data\ensemble_results.csv")
df2=pd.read_csv(r"D:\DL\HW3CNN\HW3data\single_cell_spatial&celltype.csv")
#找到有多少的种类我们需要用到unique函数，可以去掉重复值
category=df2["cell_type"].unique()
print(category)
# 发现有9种细胞
# 分别是['Normal Epithelial' 'Cancer Epithelial' 'Plasmablasts' 'PVL' 'Myeloid''B-cells' 'Endothelial' 'T-cells' 'CAFs']
# 接着是进行匹配，利用df1的坐标匹配df2的属于哪个spot内
SPOT_SIDE = 256          # Spot边长
HALF_SIDE = SPOT_SIDE / 2  # 正方形半边长
SPOT_CSV_PATH = r"D:\DL\HW3CNN\HW3data\coor.csv"
CELL_CSV_PATH = r"D:\DL\HW3CNN\HW3data\single_cell_spatial&celltype.csv"
spot_df = pd.read_csv(SPOT_CSV_PATH, index_col=0)  # 第一列是spot名称(spot_0/1...)
spot_centers = spot_df[['x', 'y']].values  # 转为数组用于计算
print(f" 成功加载 {len(spot_df)} 个Spot中心坐标")

# 读取细胞坐标+类型数据
cell_df = pd.read_csv(CELL_CSV_PATH)
# 提取细胞坐标
cell_coords = cell_df[['x', 'y']].values
print(f" 成功加载 {len(cell_df)} 个细胞数据")

#3. 极速匹配：KDTree + 正方形判断
# 构建空间索引（毫秒级）
tree = KDTree(spot_centers)
# 为每个细胞找到【最近的1个Spot】
nearest_spot_ids = tree.query(cell_coords, k=1)[1]
# 提取对应Spot的中心坐标
nearest_spots = spot_centers[nearest_spot_ids]

# 向量化计算：正方形范围判断
dx = np.abs(cell_coords[:, 0] - nearest_spots[:, 0])
dy = np.abs(cell_coords[:, 1] - nearest_spots[:, 1])
in_spot = (dx <= HALF_SIDE) & (dy <= HALF_SIDE)

#4. 标注细胞所属Spot
# 生成结果：匹配成功=Spot名称，失败=未匹配
cell_df['belong_spot'] = np.where(
    in_spot,
    spot_df.index[nearest_spot_ids],  # 用spot_0/spot_1命名
    "未匹配"
)

#5. 统计每个Spot的细胞数量
# 筛选匹配成功的细胞，统计数量
spot_cell_count = cell_df[cell_df['belong_spot'] != "未匹配"]['belong_spot'].value_counts().sort_index()
# 转为DataFrame，方便查看和保存
count_result = pd.DataFrame({
    'spot_id': spot_cell_count.index,
    'cell_count': spot_cell_count.values
})

#6. 输出结果 + 保存文件
print("每个Spot的细胞数量统计：")
print(count_result.to_string(index=False))
print("="*50)

# 保存带归属的细胞数据（可选，方便后续分析）
cell_df.to_csv(r"D:\DL\HW3CNN\HW3data\细胞归属Spot结果.csv", index=False, encoding='utf-8-sig')
# 保存Spot细胞数量统计
count_result.to_csv(r"D:\DL\HW3CNN\HW3data\Spot细胞数量统计.csv", index=False, encoding='utf-8-sig')

print("\n处理完成！")
print("1. 细胞归属结果已保存：细胞归属Spot结果.csv")
print("2. 数量统计已保存：Spot细胞数量统计.csv")
print(f"总匹配成功细胞数：{len(cell_df[cell_df['belong_spot'] != '未匹配'])} 个")
print(f"未匹配细胞数：{len(cell_df[cell_df['belong_spot'] == '未匹配'])} 个")




df=pd.read_csv(r"D:\DL\HW3CNN\HW3data\Spot细胞数量统计.csv")
# 筛选出任意一列有缺失值的行
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)
# 缺失值比例（百分比）
print((df.isnull().sum() / len(df)) * 100)
print(df.iloc[1:20])



x_train = []  
x_val = []
x_test = []
trian_folder_path = r"D:\DL\HW3CNN\patches_256_train"  # 图片所在文件夹
val_folder_path = r"D:\DL\HW3CNN\patches_256_val"
test_folder_path = r"D:\DL\HW3CNN\patches_256_test"
for path in [trian_folder_path,val_folder_path,test_folder_path]:
    for file_name in os.listdir(path):
        img_path = os.path.join(path, file_name)
        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        if path == trian_folder_path:
            x_train.append(img_tensor)
        elif path == val_folder_path:
            x_val.append(img_tensor)
        else:
            x_test.append(img_tensor)
    # Image 是python自带的库，PIL库中的一个类，可以用来处理图像数据。通过Image.open()函数可以打开一个图像文件，并返回一个Image对象
    # 这个对象包含了图像的像素数据和一些相关信息，可以进行各种图像处理操作，比如裁剪、旋转、调整大小等
    # 在这里，我们使用Image.open()函数打开了一个名为"spot_0.png"的图像文件，并将其赋值给变量img。
    # 最后，我们打印了这个Image对象，以查看它的内容和属性。
    # to_tensor=transforms.ToTensor()
    #这一步是实例化，可以看到在ToTensor类中定义了__init__,发现里面没有需要传的参数，于是实例化后自动调用__call__方法
    # 这个方法使得实例化对象可以像函数一样被调用。当我们创建了to_tensor对象后，我们可以直接使用to_tensor(img)来将PIL图像转换为PyTorch张量。
    # img_tensor=to_tensor(img)
    # print(img_tensor.shape)  # 输出图像的张量形状，应该是 [3, 256, 256]，表示3个通道（RGB）和256x256的图像尺寸
    # tensorboard_writer = SummaryWriter(log_dir="./logs")
    # tensorboard_writer.add_image(r"D:\DL\HW3CNN\patches_256\spot_0.png", img_tensor, 0) 
    # tensorboard_writer.close()
    # 这时候新建了一个log的日志，如何查看呢，需要在终端输入tensorboard --logdir=./logs
    # 这时候会提示你访问地址，通常是http://localhost:6006
    print(os.listdir(trian_folder_path))
    print(type(os.listdir(val_folder_path)))
    # 接下来处理y_train,y_val,y_test
    y_train = []
    y_val = []  
    y_test = []
    y=pd.read_csv(r"D:\DL\HW3CNN\HW3data\Spot细胞数量统计.csv")
    num_train=[]
    num_val=[]
    num_test=[]
    # 一般来说读表格就是用pandas库，pandas库中的read_csv函数可以用来读取CSV文件，并将其转换为一个DataFrame对象。
    # DataFrame是一种二维表格数据结构，可以方便地进行数据分析和处理。默认是把第一行作为列名，第一列作为索引，如果没有索引就会自动生成一个从0开始的整数索引。
    for path in [trian_folder_path,val_folder_path,test_folder_path]:
        train_list=[]
        val_list=[]
        test_list=[]    
        for file_name in os.listdir(path):
            spot_num=file_name.split(".")[0]  # 去掉文件扩展名,split是切开，然后返回一个列表，比如"spot_0.png"会被切成["spot_0","png"]，我们取第一个元素就是"spot_0"
            if path == trian_folder_path:
                train_list.append(spot_num)
            elif path == val_folder_path:
                val_list.append(spot_num)
            elif path == test_folder_path:
                test_list.append(spot_num)


        for i in range(len(y)):
            # 第 i 行，第 0 列（序号）→ pandas 必须用 .iloc
            spot_id = y.iloc[i, 0]  
            
            if spot_id in train_list:
                y_train.append(y.iloc[i, 1:].values)
            elif spot_id in val_list:
                y_val.append(y.iloc[i, 1:].values)
            elif spot_id in test_list:
                y_test.append(y.iloc[i, 1:].values)
                # 要注意格式的转换，pandas不能用[]来索引，需要用.iloc或者.loc，iloc是按位置索引，loc是按标签索引
                #这里我们用.iloc来按位置索引，第一列是序号，所以是0，后面的是标签，所以是1:，values是把DataFrame转换成numpy数组
save_path = r"D:\DL\HW3CNN\HW3data"

# 保存 y_train
pd.DataFrame([row.tolist() for row in y_train]).to_csv(
    os.path.join(save_path, "y_train1.csv"), index=False, header=False
)

# 保存 y_val
pd.DataFrame([row.tolist() for row in y_val]).to_csv(
    os.path.join(save_path, "y_val1.csv"), index=False, header=False
)

# 保存 y_test
pd.DataFrame([row.tolist() for row in y_test]).to_csv(
    os.path.join(save_path, "y_test1.csv"), index=False, header=False
)



print(np.mean(y_train))