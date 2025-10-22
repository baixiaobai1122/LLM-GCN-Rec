# 快速开始指南 - Phase 1 双图LightGCN

## 🚀 一键运行

```bash
cd /home/chuc0007/reccase/LightGCN-PyTorch/code

# 方式1：运行完整pipeline（推荐）
python run_phase1.py --dataset amazon-book --epochs 100

# 方式2：运行快速测试（5个epoch验证）
bash quick_test.sh
```

---

## 📁 已创建的文件

### 核心代码（7个文件）

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `extract_clip_features.py` | CLIP特征提取 | book_metadata.json | clip_features.npy |
| `build_semantic_graph.py` | Faiss构建语义图 | clip_features.npy | semantic_graph.npz |
| `dual_graph_model.py` | 双图LightGCN模型 | - | 模型类 |
| `dual_graph_dataloader.py` | 数据加载器 | - | 数据加载器类 |
| `train_dual_graph.py` | 训练脚本 | 图+数据 | 模型权重 |
| `run_phase1.py` | 端到端Pipeline | - | 完整流程 |
| `analyze_semantic_graph.py` | 语义图分析 | semantic_graph.npz | 统计+可视化 |

### 辅助文件

- `quick_test.sh`: 快速测试脚本
- `README_PHASE1.md`: 详细文档
- `QUICKSTART.md`: 本文件

---

## 🔄 标准工作流

### Step 0: 检查数据

```bash
cd /home/chuc0007/reccase/LightGCN-PyTorch/data/amazon-book

# 确保存在以下文件
ls -lh train.txt test.txt item_list.txt book_metadata.json

# 如果缺少book_metadata.json，运行：
cd ../../code
python fetch_book_metadata.py
```

### Step 1: 提取CLIP特征（约5-10分钟）

```bash
cd /home/chuc0007/reccase/LightGCN-PyTorch/code

python extract_clip_features.py \
  --data_path ../data/amazon-book \
  --batch_size 32
```

**输出检查**：
```bash
ls -lh ../data/amazon-book/clip_features.npy
# 应该看到 ~400MB 文件
```

### Step 2: 构建语义图（约1-2分钟）

```bash
python build_semantic_graph.py \
  --data_path ../data/amazon-book \
  --k 10 \
  --method knn \
  --normalize symmetric
```

**输出检查**：
```bash
ls -lh ../data/amazon-book/semantic_graph.npz
# 应该看到 ~几十MB 文件
```

**验证图质量**：
```bash
python analyze_semantic_graph.py \
  --data_path ../data/amazon-book \
  --n_examples 5 \
  --plot
```

### Step 3: 训练模型（约30-60分钟，100 epochs）

```bash
python train_dual_graph.py \
  --dataset amazon-book \
  --path ../data/amazon-book \
  --epochs 100 \
  --recdim 64 \
  --layer 3 \
  --use_semantic_graph 1 \
  --semantic_weight 0.5 \
  --semantic_layers 2 \
  --tensorboard 1
```

**监控训练**：
```bash
# 另开一个终端
cd /home/chuc0007/reccase/LightGCN-PyTorch/code
tensorboard --logdir runs
# 浏览器打开 http://localhost:6006
```

---

## 🎯 常用命令

### 1. 完整Pipeline（推荐新手）

```bash
# 一键运行所有步骤
python run_phase1.py --dataset amazon-book --epochs 100

# 跳过已完成的步骤
python run_phase1.py --skip_feature_extraction --skip_graph_building
```

### 2. 超参数实验

```bash
# 实验1: 不同k值
python run_phase1.py --k 5 --comment "k5"
python run_phase1.py --k 20 --comment "k20" --skip_feature_extraction

# 实验2: 不同语义权重
python run_phase1.py --semantic_weight 0.3 --skip_feature_extraction --skip_graph_building
python run_phase1.py --semantic_weight 0.7 --skip_feature_extraction --skip_graph_building

# 实验3: 更大的嵌入维度
python run_phase1.py --embed_dim 128 --skip_feature_extraction --skip_graph_building
```

### 3. 对比实验

```bash
# Baseline: 纯协同过滤（不用语义图）
python train_dual_graph.py \
  --dataset amazon-book \
  --path ../data/amazon-book \
  --use_semantic_graph 0 \
  --epochs 100 \
  --comment "baseline-no-semantic"

# Dual-Graph: 使用语义图
python train_dual_graph.py \
  --dataset amazon-book \
  --path ../data/amazon-book \
  --use_semantic_graph 1 \
  --semantic_weight 0.5 \
  --epochs 100 \
  --comment "dual-graph-sw0.5"
```

---

## 📊 结果查看

### 训练日志

```bash
# 查看最新训练日志
tail -f /home/chuc0007/reccase/LightGCN-PyTorch/code/runs/*/events.out.tfevents.*

# 或使用TensorBoard
tensorboard --logdir code/runs
```

### 模型权重

```bash
ls -lh /home/chuc0007/reccase/LightGCN-PyTorch/code/checkpoints/
# dual-lgn-amazon-book-64.pth.tar       # 最新模型
# dual-lgn-amazon-book-64_best.pth.tar  # 最佳模型
```

### 评估指标

训练过程会输出：
- **Recall@20**: 在top-20推荐中命中测试集的比例
- **Precision@20**: top-20推荐的精确率
- **NDCG@20**: 考虑排序质量的指标

---

## 🐛 常见问题

### Q1: ImportError: No module named 'transformers'

```bash
pip install transformers torch torchvision
```

### Q2: CUDA out of memory

```bash
# 减小batch size
python extract_clip_features.py --batch_size 16

# 或使用CPU
export CUDA_VISIBLE_DEVICES=""
```

### Q3: Faiss安装问题

```bash
# CPU版本（通用）
pip install faiss-cpu

# GPU版本（需要CUDA）
pip install faiss-gpu
```

### Q4: 模型训练很慢

```bash
# 检查是否使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，检查CUDA安装
nvidia-smi
```

### Q5: book_metadata.json不存在

```bash
cd code
python fetch_book_metadata.py
# 需要等待API调用，可能需要几分钟
```

---

## 📈 性能调优

### 推荐的实验顺序

1. **Baseline**（无语义图）
   ```bash
   python train_dual_graph.py --use_semantic_graph 0
   ```

2. **k值调优**（图密度）
   ```bash
   for k in 5 10 20 50; do
     python build_semantic_graph.py --k $k --output_name semantic_graph_k${k}.npz
     python train_dual_graph.py --semantic_graph_file semantic_graph_k${k}.npz --comment k${k}
   done
   ```

3. **权重调优**（CF vs 语义平衡）
   ```bash
   for alpha in 0.1 0.3 0.5 0.7; do
     python train_dual_graph.py --semantic_weight $alpha --comment sw${alpha}
   done
   ```

4. **架构调优**（层数、维度）
   ```bash
   python train_dual_graph.py --layer 4 --semantic_layers 3 --recdim 128
   ```

### 预期性能提升

相比baseline LightGCN：
- ✅ 冷启动物品：+10-20% Recall
- ✅ 整体Recall@20：+2-5%
- ✅ 长尾覆盖率提升

---

## 🔬 数据集统计

Amazon-Book数据集（当前）：
- 用户数：52,643
- 物品数：91,599
- 训练交互：2,380,730
- 测试交互：238,085
- 稀疏度：0.0495%

---

## 📚 进阶资源

- 详细文档：`README_PHASE1.md`
- 原版LightGCN代码：`code/main.py`
- 语义图分析：`python analyze_semantic_graph.py --help`

---

## 💡 小贴士

1. **首次运行**：使用`quick_test.sh`快速验证环境
2. **GPU加速**：确保安装`faiss-gpu`和`torch-cuda`
3. **磁盘空间**：确保有至少5GB空闲空间
4. **内存要求**：建议至少16GB RAM
5. **保存实验**：使用`--comment`参数标记不同实验

---

## 🆘 获取帮助

```bash
# 查看各脚本的帮助
python extract_clip_features.py --help
python build_semantic_graph.py --help
python train_dual_graph.py --help
python run_phase1.py --help
```

---

**祝实验顺利！🎉**
