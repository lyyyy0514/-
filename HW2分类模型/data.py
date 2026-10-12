import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 数据统计分析函数（核心：打印所有数据详情） =====================
def analyze_npz_data(data, data_name):
    """
    分析npz数据：样本量、特征维度、细胞类型分布、蛋白数量、特征统计值
    """
    print(f"{'='*50}")
    print(f"📊 【{data_name}】数据详细统计分析")
    print(f"{'='*50}")
    
    # 1. 基础数据提取
    embedding = data['embedding']
    centroids = data['centroids']
    celltype = data['celltype']
    proteins = data['proteins']
    protein_names = data['protein_names']
    
    # 2. 样本量统计
    num_cells = embedding.shape[0]  # 总细胞数（样本数）
    print(f"✅ 总细胞数量（样本量）: {num_cells} 个")
    
    # 3. 特征维度统计
    print(f"✅ Embedding特征维度: {embedding.shape}")
    print(f"✅ 细胞空间坐标(centroids)维度: {centroids.shape}")
    print(f"✅ 蛋白表达矩阵维度: {proteins.shape}")
    print(f"✅ 蛋白标志物数量: {len(protein_names)} 个")
    
    # 4. 细胞类型统计
    unique_types, type_counts = np.unique(celltype, return_counts=True)
    print(f"✅ 细胞类型总数: {len(unique_types)} 种")
    print("📌 各细胞类型样本数量：")
    for t, c in zip(unique_types, type_counts):
        print(f"   - {t}: {c} 个")
    
    # 5. 特征数值统计（均值/标准差）
    print(f"\n📌 Embedding特征统计：")
    print(f"   均值: {np.mean(embedding):.4f} | 标准差: {np.std(embedding):.4f}")
    print(f"📌 细胞坐标(centroids)统计：")
    print(f"   X坐标均值: {np.mean(centroids[:,0]):.2f} | Y坐标均值: {np.mean(centroids[:,1]):.2f}")
    print(f"   X坐标标准差: {np.std(centroids[:,0]):.2f} | Y坐标标准差: {np.std(centroids[:,1]):.2f}")
    print(f"{'='*50}\n")
    
    return num_cells, len(unique_types)

# ===================== 3. 神经网络模型定义 =====================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout_rate=0.1):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# ===================== 4. 训练/评估/预测工具函数 =====================
def train(model, train_loader, val_x, val_y, num_epochs, learning_rate, optimizer_type='adam', patience=30):
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss_list, val_loss_list = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
        val_loss_list.append(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ 早停：Epoch {epoch+1}")
                break
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
    
    model.load_state_dict(torch.load("best_model.pth"))
    return train_loss_list, val_loss_list

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return accuracy_score(y.cpu().numpy(), pred.cpu().numpy())

def predict(model, x):
    """获取预测标签"""
    model.eval()
    with torch.no_grad():
        return model(x).argmax(dim=1).cpu().numpy()

# ===================== 5. 可视化函数（损失曲线+混淆矩阵+细胞空间分布图） =====================
def plot_loss_curve(train_loss, val_loss, title):
    plt.figure(figsize=(10,4))
    plt.plot(train_loss, label='训练损失', color='#2E86AB')
    plt.plot(val_loss, label='验证损失', color='#A23B72')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} - 混淆矩阵')
    plt.ylabel('真实类型')
    plt.xlabel('预测类型')
    plt.show()

def plot_cell_spatial(centroids, labels, title):
    """
    核心：根据细胞坐标绘制空间分布图（颜色=细胞类型）
    centroids: 细胞坐标 [N,2]
    labels: 细胞类型标签 [N]
    """
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(centroids[:,0], centroids[:,1], c=labels, cmap='tab10', s=5, alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.colorbar(scatter, label='细胞类型')
    plt.grid(alpha=0.2)
    plt.show()

# ===================== 6. 贝叶斯优化目标函数 =====================
def bayesian_objective(hidden_size1, hidden_size2, learning_rate, dropout, optimizer_idx):
    input_size = x_train1.shape[1]
    output_size = len(np.unique(y_train1))
    h1 = int(hidden_size1)
    h2 = int(hidden_size2)
    model = NeuralNetwork(input_size, h1, h2, 64, output_size, dropout).to(device)
    opt = 'adam' if optimizer_idx>0.5 else 'sgd'
    train_loss, val_loss = train(model, loader1, x_val1, y_val1, 200, learning_rate, opt, 20)
    return -max(val_loss)

# ===================== 7. 主流程：数据加载+分析+训练+可视化 =====================
if __name__ == "__main__":
    # -------- 路径配置 --------
    gigatime_train_path = r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_train.npz"
    gigatime_val_path = r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_gigatime_val.npz"
    vit_train_path = r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_train.npz"
    vit_val_path = r"D:\数据挖掘\HW1 线性模型数据\HW1 线性模型数据\C02_vit_val.npz"

    # ======================================
    # 一、处理 GigaTIME 数据集（含数据统计）
    # ======================================
    print("🔴 开始处理 GigaTIME 数据集")
    data1 = np.load(gigatime_train_path, allow_pickle=True)
    data3 = np.load(gigatime_val_path, allow_pickle=True)
    
    # 📌 数据统计分析（核心：你要的均值、样本量、细胞类型数量）
    analyze_npz_data(data1, "GigaTIME 训练集")
    analyze_npz_data(data3, "GigaTIME 验证集")
    
    # 数据预处理
    embedding = data1['embedding']
    centroids = data1['centroids']
    celltype = data1['celltype']
    
    # 特征拼接 + 标准化
    x_train1 = np.concatenate((embedding, centroids), axis=1)
    y_train1_label = celltype.flatten()
    label_to_index = {lbl:i for i,lbl in enumerate(np.unique(y_train1_label))}
    y_train1 = np.array([label_to_index[lbl] for lbl in y_train1_label])
    
    mean, std = x_train1.mean(axis=0), x_train1.std(axis=0)
    x_train1 = (x_train1 - mean) / std
    
    # 验证集预处理
    emb_val, cen_val = data3['embedding'], data3['centroids']
    x_val1 = np.concatenate((emb_val, cen_val), axis=1)
    x_val1 = (x_val1 - mean) / std
    y_val1_label = data3['celltype'].flatten()
    y_val1 = np.array([label_to_index[lbl] for lbl in y_val1_label])
    
    # 转为Tensor
    x_train1 = torch.tensor(x_train1, dtype=torch.float32).to(device)
    y_train1 = torch.tensor(y_train1, dtype=torch.long).to(device)
    x_val1 = torch.tensor(x_val1, dtype=torch.float32).to(device)
    y_val1 = torch.tensor(y_val1, dtype=torch.long).to(device)
    
    # 数据加载器
    dataset1 = TensorDataset(x_train1, y_train1)
    loader1 = DataLoader(dataset1, batch_size=256, shuffle=True)
    
    # 模型训练
    input_size = x_train1.shape[1]
    output_size = len(label_to_index)
    model1 = NeuralNetwork(input_size, 256, 128, 64, output_size, 0.1).to(device)
    train_loss1, val_loss1 = train(model1, loader1, x_val1, y_val1, 1000, 0.01, 'adam', 30)
    
    # 评估 + 预测
    acc1 = evaluate(model1, x_val1, y_val1)
    y_pred1 = predict(model1, x_val1)
    y_true1 = y_val1.cpu().numpy()
    cen_val1 = data3['centroids']  # 验证集细胞坐标（用于空间绘图）
    
    # 可视化 GigaTIME
    plot_loss_curve(train_loss1, val_loss1, "GigaTIME 训练/验证损失曲线")
    plot_confusion_matrix(y_true1, y_pred1, "GigaTIME")
    plot_cell_spatial(cen_val1, y_true1, "GigaTIME 验证集 - 真实细胞类型空间分布")
    plot_cell_spatial(cen_val1, y_pred1, "GigaTIME 验证集 - 预测细胞类型空间分布")
    
    # 打印指标
    print(f"\n✅ GigaTIME 验证集准确率: {acc1*100:.2f}%")
    print("📊 GigaTIME 分类报告:")
    print(classification_report(y_true1, y_pred1))

    # ======================================
    # 二、处理 CellViT 数据集（含数据统计）
    # ======================================
    print("\n🟢 开始处理 CellViT 数据集")
    data2 = np.load(vit_train_path, allow_pickle=True)
    data4 = np.load(vit_val_path, allow_pickle=True)
    
    # 📌 数据统计分析
    analyze_npz_data(data2, "CellViT 训练集")
    analyze_npz_data(data4, "CellViT 验证集")
    
    # 预处理（复用标签映射，保证一致性）
    embedding2 = data2['embedding']
    centroids2 = data2['centroids']
    celltype2 = data2['celltype']
    x_train2 = np.concatenate((embedding2, centroids2), axis=1)
    y_train2_label = celltype2.flatten()
    y_train2 = np.array([label_to_index[lbl] for lbl in y_train2_label])
    
    mean2, std2 = x_train2.mean(axis=0), x_train2.std(axis=0)
    x_train2 = (x_train2 - mean2) / std2
    
    # 验证集
    emb_val2, cen_val2 = data4['embedding'], data4['centroids']
    x_val2 = np.concatenate((emb_val2, cen_val2), axis=1)
    x_val2 = (x_val2 - mean2) / std2
    y_val2_label = data4['celltype'].flatten()
    y_val2 = np.array([label_to_index[lbl] for lbl in y_val2_label])
    
    # 转为Tensor
    x_train2 = torch.tensor(x_train2, dtype=torch.float32).to(device)
    y_train2 = torch.tensor(y_train2, dtype=torch.long).to(device)
    x_val2 = torch.tensor(x_val2, dtype=torch.float32).to(device)
    y_val2 = torch.tensor(y_val2, dtype=torch.long).to(device)
    
    # 训练
    dataset2 = TensorDataset(x_train2, y_train2)
    loader2 = DataLoader(dataset2, batch_size=256, shuffle=True)
    model2 = NeuralNetwork(x_train2.shape[1], 256, 128, 64, output_size, 0.1).to(device)
    train_loss2, val_loss2 = train(model2, loader2, x_val2, y_val2, 1000, 0.01, 'adam', 30)
    
    # 评估 + 预测
    acc2 = evaluate(model2, x_val2, y_val2)
    y_pred2 = predict(model2, x_val2)
    y_true2 = y_val2.cpu().numpy()
    
    # 可视化 CellViT
    plot_loss_curve(train_loss2, val_loss2, "CellViT 训练/验证损失曲线")
    plot_confusion_matrix(y_true2, y_pred2, "CellViT")
    plot_cell_spatial(data4['centroids'], y_true2, "CellViT 验证集 - 真实细胞类型空间分布")
    plot_cell_spatial(data4['centroids'], y_pred2, "CellViT 验证集 - 预测细胞类型空间分布")
    
    print(f"\n✅ CellViT 验证集准确率: {acc2*100:.2f}%")
    print("📊 CellViT 分类报告:")
    print(classification_report(y_true2, y_pred2))