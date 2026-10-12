# 数据挖掘课程作业（Homework Repo）

> 本仓库用于存放《数据挖掘》课程的课后作业，涵盖线性模型、分类模型、CNN 图像任务等三次作业的完整代码、训练结果与可视化输出。
- **主要技术栈**：Python、NumPy、PyTorch、scikit-learn、Matplotlib、Seaborn、TensorBoard、BayesianOptimization

---

## 目录结构

```
.
├── HW1 线性模型数据/                      # 第一次作业：线性模型与神经网络实现
│   └── HW1 线性模型数据/
│       ├── ANN.py                        # 基于 NumPy 手写的前馈神经网络实现
│       ├── best_params_gigatime.json     # gigatime 任务最优超参数
│       └── best_params_vit.json          # vit 任务最优超参数
│
├── HW2分类模型/                           # 第二次作业：分类模型（多层感知机）
│   ├── HW2.py                            # 基础分类模型（3 层隐藏层 + Dropout）
│   ├── HW21.py                           # 模型改进版（调整训练结构、合并数据分布）
│   ├── HW2shuffle.py                     # 洗牌重划分版本
│   ├── data.py                           # 数据统计分析与加载
│   ├── best_model.pth                    # 最优模型权重
│   ├── best_model_final.pth              # 最终模型权重
│   └── results/
│       ├── Figure_1.png                  # 结果总览图
│       ├── 损失图/                       # 训练损失曲线
│       ├── figures/                      # 各超参数组合下的可视化
│       ├── models/                       # 各超参数组合下的模型权重
│       ├── vit分布.png / 真实分布.png     # 特征/真实分布对比
│       └── vit混淆.png                   # 混淆矩阵
│
├── 数据挖掘第三次作业/                     # 第三次作业：CNN 图像任务
│   ├── HW3CNN/
│   │   ├── cnn.py / cnn1.py              # CNN 分类模型
│   │   ├── cnn_task2.py                  # 任务2：预测每个 spot 内的细胞数量
│   │   ├── data.py / data1.py            # 数据加载与清洗
│   │   ├── dataguilei.py                 # 正六边形空间网格划分（KDTree）
│   │   ├── data_patch.py                 # 大图切块、训练/验证/测试集划分
│   │   ├── plot.py / plot1.py            # 可视化脚本
│   │   └── logs/                         # TensorBoard 训练日志
│   └── 数据挖掘第三次作业_林勇_2023215371.docx   # 作业说明文档
│
└── 数据挖掘第四次作业_林勇_2023215371.rar       # 第四次作业（压缩包）
```

---

## 各作业说明

### 1️⃣ HW1 线性模型数据

- 目标：从零实现线性模型与神经网络前向传播，理解权重、偏置、激活函数（ReLU）与矩阵化（batch）计算。
- 核心文件：`ANN.py`
- 附带：`gigatime` 与 `vit` 两组任务的最优超参数（batch_size、hidden_size、lr）保存为 JSON，供后续复用。

### 2️⃣ HW2 分类模型

- 目标：基于 PyTorch 构建多层感知机分类模型（3 层隐藏层），完成蛋白质数据分类。
- 核心要点：
  - 使用 `BayesianOptimization` 对隐藏层维度、学习率、batch size、Dropout 率进行自动调参；
  - 对 `gigatime` 与 `vit` 两类任务分别建模；
  - 将训练集与测试集合并后重新划分，保证训练/测试数据来自同一分布（考虑蛋白质方位分布）；
  - 输出训练损失曲线、特征/真实分布对比图与混淆矩阵。
- 输出：`results/` 下按 `模型_隐藏层_学习率_bs_优化器_dropout` 命名的模型权重与可视化图。

### 3️⃣ 数据挖掘第三次作业（HW3CNN）

- 目标：基于空间转录组数据（Xenium FFPE Human Breast Cancer）进行图像任务建模。
- 核心任务：
  1. **细胞分类**：读取大分辨率组织图像，切块成小 patch，构建 CNN 分类模型；
  2. **细胞计数预测**：将单细胞空间数据归类到六边形 spot 网格内，统计各类型细胞数量，建立监督回归/分类模型预测每个 spot 内的细胞数量；
  3. 使用 KDTree 完成空间坐标的六边形网格划分与单位换算。
- 配套：TensorBoard 训练日志（`logs/`）与可视化脚本。

### 4️⃣ 数据挖掘第四次作业

- 完整提交以压缩包形式保存：`数据挖掘第四次作业_林勇_2023215371.rar`。

---

## 环境依赖

| 依赖 | 用途 |
| --- | --- |
| Python 3.x | 运行环境 |
| NumPy | 数值计算 / 手写网络 |
| PyTorch | 深度学习模型构建与训练 |
| torchvision | 图像预处理与数据加载 |
| scikit-learn | 数据划分、指标计算 |
| bayes-opt | 超参数贝叶斯优化 |
| Matplotlib / Seaborn | 可视化 |
| TensorBoard | 训练日志可视化 |
| scipy（KDTree） | 空间网格划分 |
| opencv-python | 图像切块预处理 |
| tqdm | 进度条显示 |

> 安装示例：`pip install numpy torch torchvision scikit-learn bayesian-optimization matplotlib seaborn tensorboard scipy opencv-python tqdm`

---

## 快速使用

以第三次作业为例：

```bash
# 1. 数据预处理：大图切块并划分训练/验证/测试集
python data_patch.py

# 2. 空间网格划分（六边形 + 细胞归类）
python dataguilei.py

# 3. 训练 CNN 分类模型
python cnn.py

# 4. 预测每个 spot 内细胞数量
python cnn_task2.py
```

> 注意：代码中部分路径为 Windows 本地绝对路径（如 `D:\DL\...`），运行前请改为你自己的数据路径。

---

## 结果说明

- 各作业的模型权重、损失曲线、分布对比图和混淆矩阵均保存在对应作业的 `results/` 目录下。
- 超参数调优结果以 JSON 形式持久化，便于复现最佳实验配置。
- TensorBoard 日志可通过 `tensorboard --logdir logs` 查看训练过程。

---

## 注意事项

- 本仓库为课程作业存档，供学习与复现参考；
- 部分代码存在路径硬编码，跨环境运行需自行调整。
