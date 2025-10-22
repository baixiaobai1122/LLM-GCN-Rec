# Phase 1: Dual-Graph LightGCN with CLIP Features

## 概述

阶段1实现了一个快速验证方案，将CLIP提取的语义特征与协同过滤相结合。

### 核心架构

```
双图LightGCN架构：
├─ 图1：用户-物品交互图（协同过滤信号）
└─ 图2：物品-物品语义图（CLIP内容信号）

最终物品表示 = (1-α) × CF表示 + α × 语义表示
```

### 技术栈

- **特征提取**：CLIP (openai/clip-vit-base-patch32)
- **相似度搜索**：Faiss (GPU加速)
- **推荐模型**：LightGCN (扩展为双图架构)

---

## 文件结构

```
code/
├── extract_clip_features.py      # CLIP特征提取
├── build_semantic_graph.py       # Faiss构建语义图
├── dual_graph_model.py           # 双图LightGCN模型
├── dual_graph_dataloader.py      # 扩展数据加载器
├── train_dual_graph.py           # 训练脚本
└── run_phase1.py                 # 端到端Pipeline

data/amazon-book/
├── train.txt                     # 用户-物品交互
├── test.txt                      # 测试数据
├── book_metadata.json            # 书籍元数据（需先运行fetch）
├── clip_features.npy             # CLIP特征（pipeline生成）
└── semantic_graph.npz            # 语义相似度图（pipeline生成）
```

---

## 快速开始

### 1. 准备环境

```bash
cd code

# 安装依赖
pip install transformers torch torchvision faiss-cpu scipy numpy
# 如果有GPU，安装faiss-gpu以加速
# pip install faiss-gpu
```

### 2. 获取书籍元数据（如果还没有）

```bash
python fetch_book_metadata.py
```

这会生成 `../data/amazon-book/book_metadata.json`

### 3. 运行完整Pipeline

```bash
# 运行所有步骤（特征提取 + 图构建 + 训练）
python run_phase1.py --dataset amazon-book --epochs 100

# 或分步执行
python run_phase1.py --skip_training              # 只做特征提取和图构建
python run_phase1.py --skip_feature_extraction    # 跳过特征提取
```

### 4. 自定义配置

```bash
python run_phase1.py \
  --dataset amazon-book \
  --k 20 \                          # 使用top-20近邻
  --semantic_weight 0.3 \           # 语义权重30%
  --semantic_layers 3 \             # 语义图传播3层
  --epochs 200 \                    # 训练200轮
  --embed_dim 128                   # 嵌入维度128
```

---

## 分步运行

### Step 1: 提取CLIP特征

```bash
python extract_clip_features.py \
  --data_path ../data/amazon-book \
  --model_name openai/clip-vit-base-patch32 \
  --batch_size 32
```

**输出**：
- `clip_features.npy`: 物品特征矩阵 (n_items × 512)
- `book_descriptions.json`: 生成的文本描述
- `clip_features_metadata.json`: 元数据信息

**示例特征**：
```
物品0: "Title: Rachel's Holiday. Authors: Marian Keyes. Topics: Fiction, Irish, Literature..."
→ CLIP编码 → [512维向量]
```

### Step 2: 构建语义图

```bash
python build_semantic_graph.py \
  --data_path ../data/amazon-book \
  --features_file clip_features.npy \
  --method knn \
  --k 10 \
  --normalize symmetric
```

**输出**：
- `semantic_graph.npz`: 稀疏邻接矩阵 (n_items × n_items)
- `semantic_graph_metadata.json`: 图统计信息

**图构建方法**：
- `knn`: k近邻图（默认，推荐）
- `threshold`: 阈值过滤（设定最小相似度）

### Step 3: 训练双图模型

```bash
python train_dual_graph.py \
  --dataset amazon-book \
  --path ../data/amazon-book \
  --epochs 100 \
  --use_semantic_graph 1 \
  --semantic_weight 0.5 \
  --semantic_layers 2
```

**关键参数**：
- `--semantic_weight`: 语义图权重α（0.0-1.0）
  - 0.0 = 纯协同过滤
  - 1.0 = 纯内容推荐
  - 0.5 = 平衡（推荐起点）

- `--semantic_layers`: 语义图GCN层数
  - 控制语义信息传播深度

- `--recdim`: 嵌入维度（64/128/256）

**输出**：
- `checkpoints/dual-lgn-amazon-book-64.pth.tar`: 模型权重
- `runs/`: TensorBoard日志

---

## 超参数调优建议

### 1. 语义图构建

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| k | 5-50 | 近邻数量，影响图密度 |
| normalize | symmetric | 使用对称归一化（稳定） |

### 2. 模型训练

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| semantic_weight | 0.1-0.7 | 从0.3开始尝试 |
| semantic_layers | 1-3 | 2层通常足够 |
| embed_dim | 64-256 | 64是baseline |
| lr | 0.0001-0.01 | 学习率 |

### 3. 实验建议

```bash
# 实验1: 基线（无语义图）
python train_dual_graph.py --use_semantic_graph 0

# 实验2: 不同k值
for k in 5 10 20 50; do
  python build_semantic_graph.py --k $k --output_name semantic_graph_k${k}.npz
  python train_dual_graph.py --semantic_graph_file semantic_graph_k${k}.npz
done

# 实验3: 不同语义权重
for alpha in 0.1 0.3 0.5 0.7; do
  python train_dual_graph.py --semantic_weight $alpha
done
```

---

## 评估指标

训练过程会每10个epoch输出以下指标：

- **Recall@20**: 召回率（主要指标）
- **Precision@20**: 精确率
- **NDCG@20**: 归一化折损累积增益

查看训练日志：
```bash
# TensorBoard可视化
tensorboard --logdir code/runs
```

---

## 与Baseline对比

```bash
# 1. 训练原版LightGCN（baseline）
cd code
python main.py --dataset amazon-book --model lgn --epochs 100

# 2. 训练双图LightGCN
python train_dual_graph.py --dataset amazon-book --epochs 100

# 3. 比较结果
# 查看checkpoints/下的日志文件
```

**预期提升**：
- 冷启动物品：+10-20% Recall
- 整体性能：+2-5% Recall
- 长尾物品覆盖率提升

---

## 故障排查

### 问题1: CUDA out of memory

**解决方案**：
```bash
# 减小batch size
python extract_clip_features.py --batch_size 16

# 使用CPU版Faiss
pip uninstall faiss-gpu
pip install faiss-cpu
```

### 问题2: 语义图文件未找到

**错误**：`FileNotFoundError: semantic_graph.npz`

**解决方案**：
```bash
# 确保先运行前两步
python extract_clip_features.py --data_path ../data/amazon-book
python build_semantic_graph.py --data_path ../data/amazon-book
```

### 问题3: Transformers下载慢

**解决方案**：
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到cache
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

---

## 技术细节

### CLIP特征提取

- **模型**: CLIP ViT-B/32
- **输入**: 文本描述（最大77 tokens）
- **输出**: 512维归一化向量
- **相似度**: 余弦相似度

### Faiss图构建

- **索引类型**: IndexFlatIP (内积)
- **搜索**: Exact k-NN
- **归一化**: D^(-1/2) A D^(-1/2)

### 双图传播

```python
# 伪代码
user_emb, item_emb_cf = propagate_on_UI_graph(Graph_UI, n_layers=3)
item_emb_semantic = propagate_on_II_graph(Graph_II, n_layers=2)

item_emb_final = (1 - alpha) * item_emb_cf + alpha * item_emb_semantic
```

---

## 未来改进方向

1. **多模态融合**：加入书籍封面图像特征
2. **动态权重**：让α成为可学习参数
3. **注意力机制**：在双图融合时使用attention
4. **更强的文本编码器**：尝试BERT、Sentence-BERT

---

## 参考文献

- LightGCN: [He et al., SIGIR 2020]
- CLIP: [Radford et al., ICML 2021]
- Faiss: [Johnson et al., 2019]

---

## License

MIT License

## 联系方式

如有问题，请提issue或联系项目维护者。
