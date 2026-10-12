import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import os
base_dir = "./results"
os.makedirs(base_dir, exist_ok=True)

fig_dir = os.path.join(base_dir, "figures")
model_dir = os.path.join(base_dir, "models")

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# 这个代码是基于HW2.py的修改版本，主要增加了一个隐藏层，并且调整了训练函数以适应新的模型结构
# 同时重新构建了训练集测试集，因为之前的训练集和测试集是分开的
# 现在我们需要把它们合并后再进行划分，以保证是同一个分布，这考虑到了蛋白质的方位分布情况。



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
        x = torch.relu(x)
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
def plot_loss_curve(train_losses, val_losses, title, save_path=None, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    if show:
        plt.show()

    plt.close()  


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

    # 数据集1：gigatime
    print("开始处理数据集1：gigatime")

    data1 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_train.npz", allow_pickle=True)
    data3 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_val.npz", allow_pickle=True)
    data5 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_test.npz", allow_pickle=True)

    embedding_all = np.concatenate([data1['embedding'], data3['embedding'], data5['embedding']], axis=0)
    centroids_all = np.concatenate([data1['centroids'], data3['centroids'], data5['centroids']], axis=0)
    celltype_all = np.concatenate([data1['celltype'], data3['celltype'], data5['celltype']], axis=0)

    X = np.concatenate((embedding_all, centroids_all), axis=1)
    y_label = celltype_all.reshape(-1)

    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_label))}
    y = np.array([label_to_index[label] for label in y_label])

    x_train1, x_temp, y_train1, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    x_val1, x_test1, y_val1, y_test1 = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    mean = x_train1.mean(axis=0)
    std = x_train1.std(axis=0)

    x_train1 = (x_train1 - mean) / std
    x_val1 = (x_val1 - mean) / std
    x_test1 = (x_test1 - mean) / std

    x_train1 = torch.tensor(x_train1, dtype=torch.float32).to(device)
    y_train1 = torch.tensor(y_train1, dtype=torch.long).to(device)

    x_val1 = torch.tensor(x_val1, dtype=torch.float32).to(device)
    y_val1 = torch.tensor(y_val1, dtype=torch.long).to(device)

    x_test1 = torch.tensor(x_test1, dtype=torch.float32).to(device)
    y_test1 = torch.tensor(y_test1, dtype=torch.long).to(device)

    dataset1 = TensorDataset(x_train1, y_train1)
    loader1 = DataLoader(dataset1, batch_size=256, shuffle=True)

    input_size = x_train1.shape[1]
    output_size = len(np.unique(y_train1.cpu().numpy()))
    hidden_sizes = [(128, 64, 32)]
    learning_rates = [0.001]
    batch_sizes = [128]
    optimizers = ['adam']
    dropouts = [0.1]
    results_all = []
    dataset_name = "gigatime"
    for hs in hidden_sizes:
        for lr in learning_rates:
            for bs in batch_sizes:
                for opt in optimizers:
                    for dr in dropouts:

                        print(f"\n==== 当前组合 ====")
                        print(f"hidden={hs}, lr={lr}, batch={bs}, opt={opt}, dropout={dr}")

                        # DataLoader（注意batch size变化）
                        loader = DataLoader(dataset1, batch_size=bs, shuffle=True)

                        model = NeuralNetwork(
                            input_size,
                            hs[0], hs[1], hs[2],
                            output_size,
                            dropout_rate=dr
                        ).to(device)

                        train_losses, val_losses = train(
                            model,
                            loader,
                            x_val1,
                            y_val1,
                            num_epochs=300,   # 别太大，不然跑不完
                            learning_rate=lr,
                            optimizer_type=opt,
                            criterion="mse"
                        )

                        # 验证集
                        val_acc = evaluate(model, x_val1, y_val1)

                        # 测试集（关键！！）
                        test_acc = evaluate(model, x_test1, y_test1)

                        # 保存结果
                        results_all.append({
                            "dataset": dataset_name,
                            "hidden": hs,
                            "lr": lr,
                            "batch": bs,
                            "optimizer": opt,
                            "dropout": dr,
                            "val_acc": val_acc,
                            "test_acc": test_acc
                        })

                        # 保存模型（用test_acc命名）
                        model_name = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.pth"
                        fig_name   = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.png"
                        model_path = os.path.join(model_dir, model_name)
                        torch.save(model.state_dict(), model_path)

                        # 画图
                        fig_name = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.png"
                        fig_path = os.path.join(fig_dir, fig_name)

                        plot_loss_curve(
                            train_losses,
                            val_losses,
                            f"{hs}, lr={lr}, bs={bs}, {opt}, dr={dr}",
                            save_path=fig_path,
                            show=False   # 不弹窗
                        )


    # 数据集2：vit
    print("\n开始处理数据集2：vit")

    data1 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_train.npz", allow_pickle=True)
    data3 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_val.npz", allow_pickle=True)
    data5 = np.load(r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_test.npz", allow_pickle=True)

    embedding_all = np.concatenate([data1['embedding'], data3['embedding'], data5['embedding']], axis=0)
    centroids_all = np.concatenate([data1['centroids'], data3['centroids'], data5['centroids']], axis=0)
    celltype_all = np.concatenate([data1['celltype'], data3['celltype'], data5['celltype']], axis=0)

    X = np.concatenate((embedding_all, centroids_all), axis=1)
    y_label = celltype_all.reshape(-1)
    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_label))}
    y = np.array([label_to_index[label] for label in y_label])
    x_train2, x_temp, y_train2, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    x_val2, x_test2, y_val2, y_test2 = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    mean = x_train2.mean(axis=0)
    std = x_train2.std(axis=0)
    x_train2 = (x_train2 - mean) / std
    x_val2 = (x_val2 - mean) / std
    x_test2 = (x_test2 - mean) / std

    x_train2 = torch.tensor(x_train2, dtype=torch.float32).to(device)
    y_train2 = torch.tensor(y_train2, dtype=torch.long).to(device)

    x_val2 = torch.tensor(x_val2, dtype=torch.float32).to(device)
    y_val2 = torch.tensor(y_val2, dtype=torch.long).to(device)

    x_test2 = torch.tensor(x_test2, dtype=torch.float32).to(device)
    y_test2 = torch.tensor(y_test2, dtype=torch.long).to(device)

    dataset2 = TensorDataset(x_train2, y_train2)
    loader2 = DataLoader(dataset2, batch_size=256, shuffle=True)

    input_size = x_train2.shape[1]
    output_size = len(np.unique(y_train2.cpu().numpy()))
    hidden_sizes = [(128, 64, 32) ]
    learning_rates = [0.001]
    batch_sizes = [128]
    optimizers = ['adam']
    dropouts = [0.1, 0.3]
    dataset_name = "vit"
    for hs in hidden_sizes:
        for lr in learning_rates:
            for bs in batch_sizes:
                for opt in optimizers:
                    for dr in dropouts:

                        print(f"\n==== 当前组合 ====")
                        print(f"hidden={hs}, lr={lr}, batch={bs}, opt={opt}, dropout={dr}")

                        # DataLoader（注意batch size变化）
                        loader2 = DataLoader(dataset2, batch_size=bs, shuffle=True)

                        model = NeuralNetwork(
                            input_size,
                            hs[0], hs[1], hs[2],
                            output_size,
                            dropout_rate=dr
                        ).to(device)

                        train_losses, val_losses = train(
                            model,
                            loader2,
                            x_val2,
                            y_val2,
                            num_epochs=300,  
                            learning_rate=lr,
                            optimizer_type=opt,
                            criterion="mse"
                        )

                        # 验证集
                        val_acc = evaluate(model, x_val2, y_val2)

                        # 测试集（关键！！）
                        test_acc = evaluate(model, x_test2, y_test2)

                        # 保存结果
                        results_all.append({
                            "dataset": dataset_name,
                            "hidden": hs,
                            "lr": lr,
                            "batch": bs,
                            "optimizer": opt,
                            "dropout": dr,
                            "val_acc": val_acc,
                            "test_acc": test_acc
                        })

                        # 保存模型（用test_acc命名）
                        model_name = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.pth"
                        fig_name   = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.png"
                        model_path = os.path.join(model_dir, model_name)
                        torch.save(model.state_dict(), model_path)

                        # 画图
                        fig_name = f"{dataset_name}_h{hs}_lr{lr}_bs{bs}_{opt}_dr{dr}.png"
                        fig_path = os.path.join(fig_dir, fig_name)

                        plot_loss_curve(
                            train_losses,
                            val_losses,
                            f"{hs}, lr={lr}, bs={bs}, {opt}, dr={dr}",
                            save_path=fig_path,
                            show=False   # 不弹窗
                        )


