import pandas as pd
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 这个文件主要对数据进行处理，因为发现了数据有少量的绝世之，所以需要对数据进行清洗
# 首先读取两个csv文件，然后进行合并，最后查看缺失值的情况以及缺失值的比例，最后查看前20行数据的情况，以便后续处理
df1 = pd.read_csv(r"D:\DL\HW3CNN\HW3data\ensemble_results.csv")
df2=pd.read_csv(r"D:\DL\HW3CNN\HW3data\coor.csv")
df=pd.merge(df1,df2,left_on="Unnamed: 0",right_on="Unnamed: 0")
# merge函数可以将两个DataFrame对象按照指定的列进行合并
# left_on和right_on参数分别指定了左边和右边DataFrame中用于合并的列名
# 。在这里，我们使用了"Unnamed: 0"列作为合并的依据，只保留两边都有的行。合并后的DataFrame对象被赋值给变量df。
# 筛选出任意一列有缺失值的行
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)
# 缺失值比例（百分比）
print((df.isnull().sum() / len(df)) * 100)
print(df.iloc[1:20])
df.to_csv(r"D:\DL\HW3CNN\HW3data\merged_data.csv", index=False)
# to_csv函数可以将DataFrame对象保存为CSV文件


x_train = []  
x_val = []
x_test = []
trian_folder_path = r"D:\DL\HW3CNN\patches_256_train1"  # 图片所在文件夹
val_folder_path = r"D:\DL\HW3CNN\patches_256_val1"
test_folder_path = r"D:\DL\HW3CNN\patches_256_test1"
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
    y_train2 = []
    y_val2 = []  
    y_test2 = []
    y=pd.read_csv(r"D:\DL\HW3CNN\HW3data\merged_data.csv")
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
            sample_id = y.iloc[i, 0]  
            
            if sample_id in train_list:
                y_train2.append(y.iloc[i, 1:].values)
            elif sample_id in val_list:
                y_val2.append(y.iloc[i, 1:].values)
            elif sample_id in test_list:
                y_test2.append(y.iloc[i, 1:].values)
                # 要注意格式的转换，pandas不能用[]来索引，需要用.iloc或者.loc，iloc是按位置索引，loc是按标签索引
                #这里我们用.iloc来按位置索引，第一列是序号，所以是0，后面的是标签，所以是1:，values是把DataFrame转换成numpy数组
save_path = r"D:\DL\HW3CNN\HW3data"

# 保存 y_train
pd.DataFrame([row.tolist() for row in y_train2]).to_csv(
    os.path.join(save_path, "y_train.csv"), index=False, header=False
)

# 保存 y_val
pd.DataFrame([row.tolist() for row in y_val2]).to_csv(
    os.path.join(save_path, "y_val.csv"), index=False, header=False
)

# 保存 y_test
pd.DataFrame([row.tolist() for row in y_test2]).to_csv(
    os.path.join(save_path, "y_test.csv"), index=False, header=False
)



