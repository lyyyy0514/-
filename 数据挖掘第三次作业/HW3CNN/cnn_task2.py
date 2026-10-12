import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from torchvision.utils import make_grid
# 这个代码主要是完成作业的第三问，预测每个spot内的细胞数量
# 也可以使用opencv，读取得到的数据类型是ndarray，
# 使用PIL读取得到的数据类型是PIL.Image.Image，二者在后续处理时需要注意区分和转换。都可以转化为tensor
# 在python的class中，如果是下划线开头的函数（如__init__），它们被称为特殊方法或魔术方法。这些方法不需要调用方法名来使用
# 而是在调用class的时候自动调用。例如，__init__方法在创建对象时自动调用，用于初始化对象的属性。
# 调用类之前必须要实例化，实例化是给init传递参数的过程，实例化后就可以使用这个对象了。如果有默认值则可以不传递参数，直接实例化对象。
from torch.utils.tensorboard import SummaryWriter
# tensorboard是一个可视化工具，可以帮助我们更好地理解和调试深度学习模型。
# 通过SummaryWriter，我们可以将训练过程中的各种指标（如损失、准确率等）记录下来
# 并在TensorBoard中进行可视化展示。这对于监控模型的训练进展、发现潜在问题以及比较不同模型的性能非常有帮助。

device=("cuda") if torch.cuda.is_available() else("cpu")
print(f"使用设备: {device}")
writer = SummaryWriter(log_dir="./logs")

class CNN(nn.Module):
    def __init__(self,hidden_size=(512, 256),dropout_rate=0.2):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.Dropout(p=dropout_rate),#卷积层后面加一个dropout层，防止过拟合，丢弃率为0.2，可以根据需要调整
            nn.ReLU(),
            nn.BatchNorm2d(16), # 新增
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),

            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.hidden_size=hidden_size
        # 接下来是全连接层,全连接层不要太多层，两层差不多了。
        self.fc=nn.Sequential(
            nn.Linear(128, self.hidden_size[0]),#四次池化，每次池化长宽都除以2，所以16*16，每次卷积后通道数翻倍，所以128*16*16，512是隐藏层的神经元数量，可以根据需要调整
            nn.ReLU(),
            nn.Dropout(p=0.5),#全连接层的dropout率可以稍微大一点，0.5比较常见，可以根据需要调整
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(p=0.5), # 再次加入 Dropout
            nn.Linear(self.hidden_size[1], 1)#映射到9个类别上
        )
    def forward(self,x):
        #因为x输入的shape是[batch_size, 3, 256, 256]，经过四次池化后，长宽都除以2的4次方，也就是16，所以变成了[batch_size, 128, 16, 16]，展开还需要保持batch_size不变
        x=self.conv(x)
        x=x.view(x.size(0),-1)#展平，保持batch_size不变，其他维度展平
        x=self.fc(x)
        return x

def train(model,train_loaders,val_loader,num_epochs,lr):
    model.train()
    criterion = nn.MSELoss()#要预测的是数量，用MSE
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    train_losses=[]
    val_losses=[]
    best_val_loss = float('inf')#记录最佳验证损失,如果当前验证损失比最佳验证损失更小，就保存模型参数
    patience = 50 # 早停的耐心值，即连续多少个epoch验证损失没有改善就停止训练
    patience_counter = 0  # 计数器，记录连续多少个epoch验证损失没有改善

    for epoch in range(num_epochs):
        train_loss=0
        for x_batch,y_batch in train_loaders:
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device).float() 
            y_pred=model(x_batch)
            loss=criterion(y_pred,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            # 每一个batch算一次损失，所以要除去loader的长度，就是有多少个batch
        train_loss /= len(train_loaders)
        train_losses.append(train_loss)
# 一般来说这样写就可以了，但是我们的数据太大了，不能一次性把x_val和y_val输入到gpu中，所以我们需要分批次输入，计算验证损失
        # model.eval()
        # with torch.no_grad():
        #     y_pred_val=model(x_val)
        #     val_loss=criterion(y_pred_val,y_val).item()
        #     val_losses.append(val_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch,y_batch in val_loader:
                x_batch=x_batch.to(device)
                y_batch=y_batch.to(device).float() 
                y_pred=model(x_batch)
                loss=criterion(y_pred,y_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)    
        if (epoch+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        # 判断是否最好
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 可以在这里保存表现最好的模型权重
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return train_losses,val_losses
# def eval(model,x_test,y_test):
#     model.eval()
#     with torch.no_grad():
#         # 这个是专门用于计算测试集的效果的
#         logits = model(x_test)          # 模型输出 logits
#         loss = nn.CrossEntropyLoss()(logits, y_test)  # 测试损失和训练完全一样
#         probs = F.softmax(logits, dim=1)  # 计算概率
#     return loss.item(),probs

def eval(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # 1. 搬运到 GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            # 2. 前向传播
            logits = model(x_batch) # 模型输出的是 log_softmax
            
            # 3. 计算损失
            loss = criterion(logits, y_batch)
            test_loss += loss.item()



    # 计算平均损失
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss


# 验证集/测试集：仅做必要的 Resize 和 ToTensor，不做随机变换
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

#2. 数据集定义
#这个是关键，训练时需要一个 batch，才去文件夹里读几张图片，用完就释放，完全不占内存 —— 这是处理图片数据集的唯一标准写法。
#只有当数据都是tensor的时候，才可以使用TensorDataset和DataLoader，否则就需要自定义Dataset类，在__getitem__方法中读取图片并进行变换，这样就不会占用过多内存了。
class MyDataset(Dataset):
    def __init__(self, folder_path, labels, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.img_names = sorted(os.listdir(folder_path)) 
        # 直接保存传入的 labels (已经是 tensor 或 numpy)
        self.labels = labels 
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 这里确保 label 是 float32 tensor
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return image, label
    
# 每次都只拿一个图片出来，在dataloader中自动循环读取，训练完就释放，不占内存
if __name__ == "__main__":
    y_train1=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_train1.csv",header=None)
    y_val1=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_val1.csv",header=None)
    y_test1=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_test1.csv",header=None)
#必须要这么设置，不然就少一行数据
    print(y_train1.shape)
    # x_train = []  
    # x_val = []
    # x_test = []
    train_folder_path = r"D:\DL\HW3CNN\patches_256_train1"  # 图片所在文件夹
    val_folder_path = r"D:\DL\HW3CNN\patches_256_val1"
    test_folder_path = r"D:\DL\HW3CNN\patches_256_test1"
    # 现在不需要提前把所有的图片都读入内存了，我们在Dataset中直接读取图片并进行变换，这样就不会占用过多内存了
    # for path in [train_folder_path,val_folder_path,test_folder_path]:
    #     for file_name in os.listdir(path):
    #         img_path = os.path.join(path, file_name)
    #         img = Image.open(img_path)
    #         to_tensor = transforms.ToTensor()
    #         img_tensor = to_tensor(img)
    #         if path == train_folder_path:
    #             x_train.append(img_tensor)
    #         elif path == val_folder_path:
    #             x_val.append(img_tensor)
    #         else:
    #             x_test.append(img_tensor)
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
    model=CNN().to(device)
    # x_train=torch.stack(x_train)#把list中的tensor堆叠成一个大的tensor，shape是[batch_size, 3, 256, 256]
    y_train1=y_train1.to_numpy()#绝对不允许直接把所有的数据一次性输入gpu
    y_train1=torch.tensor(y_train1)#把标签转换成tensor，shape是[batch_size]，每个元素是一个整数，表示类别标签
    # x_val=torch.stack(x_val)
    y_val1=y_val1.to_numpy()
    y_val1=torch.tensor(y_val1)
    #   x_test=torch.stack(x_test)
    y_test1=y_test1.to_numpy()
    y_test1=torch.tensor(y_test1)
    # tensor没有values属性
    # print("x_train shape:", x_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("x_val shape:", x_val.shape)
    # print("y_val shape:", y_val.shape)
    # print("x_test shape:", x_test.shape)
    # print("y_test shape:", y_test.shape)


    # dataset=TensorDataset(x_train,y_train)#不是Dataset
    # dataloader=DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),        # 确保尺寸一致
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                # 转化为 Tensor 并归一化到 [0, 1]
    ])
    train_dataset = MyDataset(train_folder_path, y_train1, transform=train_transform)
    val_dataset = MyDataset(val_folder_path, y_val1, transform=test_transform)

    # 实例化 DataLoader (这会处理 batching 和 shuffle)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    dataiter = iter(train_loader)
    #对比：原图 + 增强图 
    # 1. 取 【无增强的原始图片】 (用于对比)
    original_dataset = MyDataset(train_folder_path, y_train1, transform=test_transform)
    original_loader = DataLoader(original_dataset, batch_size=32, shuffle=False)
    original_iter = iter(original_loader)
    original_images, _ = next(original_iter)

    # 2. 取 【增强后的图片】 (你原来的代码)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # 3. 分别生成网格图
    original_grid = make_grid(original_images, nrow=8)
    aug_grid = make_grid(images, nrow=8)

    # 4. 写入TensorBoard，分两个标签，方便对比
    writer.add_image('Original Images (变换前)', original_grid, global_step=0)
    writer.add_image('Augmented Images (变换后)', aug_grid, global_step=0)
    print("已上传【原图+增强图】到 TensorBoard，可直接对比！")

    train_losses,val_losses=train(model,train_loader,val_loader,num_epochs=1000,lr=0.001)
    test_dataset = MyDataset(test_folder_path, y_test1, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, #windows电脑通常设置为0，linux可以设置为更大的值来加速数据加载，因为不支持多线程，这个和之前yolov8检检测油茶的那个一样
        pin_memory=True
    )

    #2. 训练完成后，进行测试
    print("开始最终测试...")
    final_test_loss = eval(model, test_loader)
    
    print(f"最终测试损失 : {final_test_loss:.4f}")
    
    # 最后记得关掉 writer
    writer.close()

