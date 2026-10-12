import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
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
            nn.MaxPool2d(2,2)
        )
        self.hidden_size=hidden_size
        # 接下来是全连接层,全连接层不要太多层，两层差不多了。
        self.fc=nn.Sequential(
            nn.Linear(128 * 16 * 16, self.hidden_size[0]),#四次池化，每次池化长宽都除以2，所以16*16，每次卷积后通道数翻倍，所以128*16*16，512是隐藏层的神经元数量，可以根据需要调整
            nn.ReLU(),
            nn.Dropout(p=0.5),#全连接层的dropout率可以稍微大一点，0.5比较常见，可以根据需要调整
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Dropout(p=0.5), # 再次加入 Dropout
            nn.Linear(self.hidden_size[1], 9)#映射到9个类别上
        )
    def forward(self,x):
        #因为x输入的shape是[batch_size, 3, 256, 256]，经过四次池化后，长宽都除以2的4次方，也就是16，所以变成了[batch_size, 128, 16, 16]，展开还需要保持batch_size不变
        x=self.conv(x)
        x=x.view(x.size(0),-1)#展平，保持batch_size不变，其他维度展平
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

def train(model,loaders,x_val,y_val,num_epochs,lr):
    model.train()
    criterion = nn.KLDivLoss(reduction='batchmean')#要预测的是概率之差，应该使用KL散度损失函数
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    train_losses=[]
    val_losses=[]
    best_val_loss = float('inf')#记录最佳验证损失,如果当前验证损失比最佳验证损失更小，就保存模型参数
    patience = 30  # 早停的耐心值，即连续多少个epoch验证损失没有改善就停止训练
    patience_counter = 0  # 计数器，记录连续多少个epoch验证损失没有改善

    for epoch in range(num_epochs):
        train_loss=0
        for x_batch,y_batch in loaders:
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device).float() 
            y_pred=model(x_batch)
            loss=criterion(y_pred,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            # 每一个batch算一次损失，所以要除去loader的长度，就是有多少个batch
        train_loss /= len(loaders)
        train_losses.append(train_loss)
# 一般来说这样写就可以了，但是我们的数据太大了，不能一次性把x_val和y_val输入到gpu中，所以我们需要分批次输入，计算验证损失
        # model.eval()
        # with torch.no_grad():
        #     y_pred_val=model(x_val)
        #     val_loss=criterion(y_pred_val,y_val).item()
        #     val_losses.append(val_loss)
        model.eval()
        val_loss = 0
        val_batch_size = 32
        num_bataches = (len(x_val) + val_batch_size - 1) // val_batch_size  # 计算验证集的批次数
        with torch.no_grad():
            for i in range(0, len(x_val), val_batch_size):
                x_v = x_val[i:i+val_batch_size].to(device)
                y_v = y_val[i:i+val_batch_size].to(device).float()
                pred_v = model(x_v)
                val_loss += criterion(pred_v, y_v).item()
        
        val_loss /= num_bataches
        val_losses.append(val_loss)
        if (epoch+1) % 50 == 0:
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

def eval(model, x_test, y_test):
    model.eval()
    criterion = nn.KLDivLoss(reduction='batchmean')

    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    test_loss = 0
    all_probs = []

    with torch.no_grad():
        for x_t, y_t in loader:
            x_t = x_t.to(device)
            y_t = y_t.to(device).float()

            logits = model(x_t)
            loss = criterion(logits, y_t)

            test_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    test_loss /= len(loader)
    all_probs = torch.cat(all_probs, dim=0)

    return test_loss, all_probs


if __name__ == "__main__":
    y_train=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_train.csv",header=None).iloc[:, :-2]
    y_val=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_val.csv",header=None).iloc[:, :-2]
    y_test=pd.read_csv(r"D:\DL\HW3CNN\HW3data\y_test.csv",header=None).iloc[:, :-2]
#必须要这么设置，不然就少一行数据
    print(y_train.shape)
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
    model=CNN().to(device)
    x_train=torch.stack(x_train)#把list中的tensor堆叠成一个大的tensor，shape是[batch_size, 3, 256, 256]
    y_train=y_train.to_numpy()#绝对不允许直接把所有的数据一次性输入gpu
    y_train=torch.tensor(y_train)#把标签转换成tensor，shape是[batch_size]，每个元素是一个整数，表示类别标签
    x_val=torch.stack(x_val)
    y_val=y_val.to_numpy()
    y_val=torch.tensor(y_val)
    x_test=torch.stack(x_test)
    y_test=y_test.to_numpy()
    y_test=torch.tensor(y_test)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)


    dataset=TensorDataset(x_train,y_train)#不是Dataset
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True)
    train_losses,val_losses=train(model,dataloader,x_val,y_val,num_epochs=100,lr=0.001)
    test_loss,probs=eval(model,x_test,y_test)
    print(f"测试损失: {test_loss}")
    print(f"测试集预测的概率: {probs}")

