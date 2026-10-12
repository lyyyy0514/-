import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from bayes_opt import BayesianOptimization
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，也可以换成其他中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3, output_size,dropout_rate=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3=hidden_size3
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self. hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.fc4=nn.Linear(self.hidden_size3,output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x=self.dropout(x)
        x=self.fc4(x)
        return x


def train(model,loader, X_val,y_val, num_epochs=1000, learning_rate=0.01,optimizer_type='adam',criterion="crossentropyloss",patience=20):
    if criterion=="crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif criterion == "mse":
        criterion = nn.MSELoss()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')   
    patience_counter = 0           
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch,y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(loader)
        train_losses.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val).item()
            val_losses.append(val_loss)
            # Early Stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    return train_losses, val_losses
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy
def plot_loss_curve(train_losses, val_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def bayesian_objective(hidden_size1, hidden_size2, learning_rate, dropout, optimizer_idx):
    hidden_size1 = int(hidden_size1)
    hidden_size2 = int(hidden_size2)

    optimizer_type = 'adam' if optimizer_idx > 0.5 else 'sgd'

    input_size = x_train1.shape[1]
    output_size = len(np.unique(y_train1))

    model = NeuralNetwork(
        input_size,
        hidden_size1,
        hidden_size2,
        output_size,
        dropout_rate=dropout
    ).to(device)

    train(
        model,
        loader1,
        x_val1,
        y_val1,
        num_epochs=200,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        patience=15
    )

    acc = evaluate(model, x_val1, y_val1)

    return acc


if __name__ == "__main__":
    # 数据集路径
    data1=np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_train.npz",allow_pickle=True)
    data3=np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_val.npz",allow_pickle=True)
    # 要打开pickle文件需要加上allow_pickle=True参数，否则会报错
    print(data1.files)
    embedding=data1['embedding']
    centroids=data1['centroids']
    protein_names=data1['protein_names']
    proteins=data1['proteins']
    print(f"proteins.shape: {proteins.shape}")
    celltype=data1['celltype']
    celltype = celltype.reshape(-1, 1)
    print(f"celltype.shape: {celltype.shape}")
    x_train1=np.concatenate((embedding,centroids),axis=1)
    print(f"celltype[:5]: {celltype[:5]}")
    y_train1_label=celltype.flatten()  # 将二维数组展平为一维数组
    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_train1_label))}
    y_train1 = np.array([label_to_index[label] for label in y_train1_label])
    mean = x_train1.mean(axis=0)
    std = x_train1.std(axis=0)
    x_train1 = (x_train1 - mean) / std
    dataset1 = TensorDataset(torch.tensor(x_train1, dtype=torch.float32).to(device), torch.tensor(y_train1, dtype=torch.long).to(device))
    loader1 = DataLoader(dataset1, batch_size=256, shuffle=True)
    print(x_train1.shape)
    print(y_train1.shape)
    print(y_train1[:5])
    print(x_train1.std(axis=0))
    embedding=data3['embedding']
    centroids=data3['centroids']
    protein_names=data3['protein_names']
    proteins=data3['proteins']
    celltype=data3['celltype']
    celltype = celltype.reshape(-1, 1)
    print(celltype.shape)
    x_val1=np.concatenate((embedding,centroids),axis=1)
    x_val1=torch.tensor(x_val1,dtype=torch.float32).to(device)
    y_val1_label=celltype.flatten()  # 将二维数组展平为一维数组
    y_val1 = np.array([label_to_index[label] for label in y_val1_label])
    y_val1=torch.tensor(y_val1, dtype=torch.long).to(device)
    x_val1 = x_val1.cpu().numpy()
    x_val1 = (x_val1 - mean) / std
    x_val1 = torch.tensor(x_val1, dtype=torch.float32).to(device)
    print(x_val1.shape)
    print(y_val1.shape)
    print(y_val1[:5])
    data2=np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_train.npz",allow_pickle=True)
    data4=np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_val.npz",allow_pickle=True)
    # 要打开pickle文件需要加上allow_pickle=True参数，否则会报错
    print(data2.files)
    embedding2=data2['embedding']
    centroids2=data2['centroids']
    protein_names2=data2['protein_names']
    proteins2=data2['proteins']
    celltype2=data2['celltype']
    celltype2 = celltype2.reshape(-1, 1)
    print(celltype2.shape)
    x_train2=np.concatenate((embedding2,centroids2),axis=1)
    y_train2_label=celltype2.flatten()  # 将二维数组展平为一维数组
    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_train2_label))}
    y_train2 = np.array([label_to_index[label] for label in y_train2_label])
    print(x_train2.shape)
    print(y_train2.shape)
    print(y_train2[:5])
    print(x_train2.std(axis=0))
    embedding2=data2['embedding']
    centroids2=data2['centroids']
    protein_names2=data2['protein_names']
    proteins2=data2['proteins']
    celltype2=data2['celltype']
    celltype2 = celltype2.reshape(-1, 1)
    print(celltype2.shape)
    x_val2=np.concatenate((embedding2,centroids2),axis=1)
    y_val2_label=celltype2.flatten()  # 将二维数组展平为一维数组
    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_val2_label))}
    y_val2 = np.array([label_to_index[label] for label in y_val2_label])




    #训练第一个数据集：gigatime
    print("开始处理数据集1：gigatime")
    # best_params1 = bayesian_optimize(x_train1, x_val1, y_train1, y_val1)
    # 用最优参数训练最终模型
    pbounds = {
    'hidden_size1': (32, 256),
    'hidden_size2': (16, 128),
    'learning_rate': (1e-4, 0.05),
    'dropout': (0.1, 0.5),
    'optimizer_idx': (0, 1)   # 0=SGD, 1=Adam
        }
    optimizer = BayesianOptimization(
    f=bayesian_objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2)
    optimizer.maximize(init_points=10,n_iter=30)
    best_params = optimizer.max['params']
    best_h1 = int(best_params['hidden_size1'])
    best_h2 = int(best_params['hidden_size2'])
    best_lr = best_params['learning_rate']
    best_dropout = best_params['dropout']
    best_optimizer = 'adam' if best_params['optimizer_idx'] > 0.5 else 'sgd'
    print("最优参数：", best_params)
    # 把最佳参数保存下来
    input_size = x_train1.shape[1]
    output_size = len(np.unique(y_train1))#输出类别数，取所有的蛋白质名称的数量
    model1 = NeuralNetwork(input_size,256,128,64,output_size,dropout_rate=0.1).to(device)
    train_loss1, val_loss1 = train(model1,loader1,x_val1,y_val1,num_epochs=1000,learning_rate=0.01,optimizer_type='adam',patience=30)
    torch.save(model1.state_dict(), "best_model_final.pth")
    plot_loss_curve(train_loss1, val_loss1, "Gigatime Loss Curve")
    accuracy1 = evaluate(model1, x_val1, y_val1)
    print(f"数据集1：gigatime 的验证集准确率为: {accuracy1 * 100:.2f}%")

