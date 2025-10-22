# Project Restructuring Summary

## Overview
Restructured `LightGCN-PyTorch` → **`LLM-GCN-Rec`** for better organization and maintainability.

**Date**: October 21, 2025
**Backup**: `code_backup_20251021_141257/`

---

## 🆕 New Project Structure

```
LLM-GCN-Rec/
├── src/                          # Source code
│   ├── baseline/                 # Original LightGCN implementation
│   │   ├── model.py             # LightGCN model
│   │   ├── dataloader.py        # Data loading
│   │   ├── Procedure.py         # Training procedures
│   │   ├── utils.py             # Utilities (BPRLoss, etc.)
│   │   ├── world.py             # Configuration
│   │   ├── register.py          # Model registration
│   │   └── parse.py             # Argument parsing
│   │
│   ├── models/                   # Enhanced models
│   │   ├── dual_graph_model.py  # Dual-graph LightGCN
│   │   └── dual_graph_dataloader.py
│   │
│   ├── data/                     # Data processing
│   │   ├── create_subset.py     # Dataset subsetting
│   │   ├── fetch_book_metadata.py
│   │   └── semantic_graph/      # Semantic graph construction
│   │       ├── build_graph.py
│   │       ├── extract_features.py
│   │       └── analyze_graph.py
│   │
│   ├── training/                 # Training scripts
│   │   ├── train_baseline.py    # Baseline LightGCN training
│   │   ├── train_dualgraph.py   # Dual-graph training
│   │   └── run_phase1.py
│   │
│   ├── evaluation/               # Evaluation (future)
│   └── utils/                    # Common utilities (future)
│
├── experiments/                  # Experiment scripts
│   └── grid_search/
│       ├── quick_grid_search.py
│       └── grid_search_dualgraph.py
│
├── datasets/                     # Data (renamed from 'data/')
│   ├── amazon-book/
│   ├── amazon-book_subset_1500/
│   ├── gowalla/
│   ├── lastfm/
│   └── yelp2018/
│
├── logs/                         # Logs (renamed from 'log/')
│   ├── baseline/
│   ├── dualgraph/
│   └── grid_search/
│
├── checkpoints/                  # Model checkpoints
│   ├── baseline/
│   └── dualgraph/
│
├── results/                      # Experiment results
│   ├── baseline/
│   ├── dualgraph/
│   └── grid_search/
│
├── configs/                      # Configuration files (future)
├── scripts/                      # Shell scripts (future)
├── notebooks/                    # Jupyter notebooks (future)
└── docs/                         # Documentation (future)
```

---

## 📝 Key Changes

### 1. **Directory Restructuring**
- `code/` → Split into `src/baseline/`, `src/models/`, `src/training/`, `src/data/`
- `data/` → `datasets/`
- `log/` → `logs/` with subdirectories
- Created `experiments/` for grid search and ablation studies

### 2. **Import Path Updates**
All import statements updated to use new structure:

**Baseline modules** (`src/baseline/`):
- Imports within baseline use relative imports
- Updated paths: `ROOT_PATH`, `DATA_PATH`, `FILE_PATH`

**Dual-graph models** (`src/models/`):
```python
from src.baseline import world
from src.baseline.dataloader import Loader
```

**Training scripts** (`src/training/`):
```python
from src.baseline import world, utils, Procedure, register
from src.models.dual_graph_dataloader import DualGraphLoader
from src.models.dual_graph_model import DualGraphLightGCN
```

**Grid search** (`experiments/grid_search/`):
```python
train_script = '../../src/training/train_dualgraph.py'
```

### 3. **Path References Updated**
- Dataset paths: `../../datasets/amazon-book_subset_1500`
- Log directories: `logs/baseline/`, `logs/dualgraph/`
- Checkpoints: `checkpoints/baseline/`, `checkpoints/dualgraph/`
- Results: `results/grid_search/`

---

## 🚀 How to Use

### Training Baseline LightGCN
```bash
cd /home/chuc0007/reccase/LLM-GCN-Rec

python src/training/train_baseline.py \
    --dataset amazon-book_subset_1500 \
    --path datasets/amazon-book_subset_1500 \
    --epochs 100
```

### Training Dual-Graph Model
```bash
python src/training/train_dualgraph.py \
    --dataset amazon-book_subset_1500 \
    --path datasets/amazon-book_subset_1500 \
    --epochs 100 \
    --semantic_weight 0.5 \
    --semantic_layers 2
```

### Running Grid Search
```bash
cd experiments/grid_search
python quick_grid_search.py
```

---

## 🔄 Migration Guide

If you have custom scripts using the old structure:

### Old → New Import Paths

| Old | New |
|-----|-----|
| `import world` | `from src.baseline import world` |
| `import model` | `from src.baseline import model` |
| `from utils import BPRLoss` | `from src.baseline.utils import BPRLoss` |
| `import Procedure` | `from src.baseline import Procedure` |
| `from dual_graph_model import ...` | `from src.models.dual_graph_model import ...` |

### Old → New File Paths

| Old | New |
|-----|-----|
| `../data/amazon-book` | `datasets/amazon-book` |
| `code/checkpoints/` | `checkpoints/baseline/` or `checkpoints/dualgraph/` |
| `code/runs/` | `runs/` |
| `log/` | `logs/baseline/` or `logs/dualgraph/` |

---

## ✅ Verified Components

- ✅ Project renamed to `LLM-GCN-Rec`
- ✅ Folder structure created
- ✅ Files moved to appropriate locations
- ✅ Import paths updated in all files
- ✅ Path references updated
- ✅ Grid search scripts updated
- ⏳ Testing in progress

---

## 📦 Backup

Original code backed up at:
```
/home/chuc0007/reccase/LLM-GCN-Rec/code_backup_20251021_141257/
```
Size: 8.0 MB

---

## 🐛 Known Issues & TODO

- [ ] Test baseline training with new structure
- [ ] Test dual-graph training with new structure
- [ ] Add configuration file support (`configs/`)
- [ ] Create shell scripts for common operations (`scripts/`)
- [ ] Add comprehensive documentation

---

## 📞 Questions?

If you encounter any issues with the new structure, check:
1. Import paths are correct
2. Dataset paths point to `datasets/` not `data/`
3. Working directory is project root when running scripts
4. Python path includes project root
