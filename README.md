# OmniScale — High-Performance Neural Recommender & Logistics Optimizer

<p align="center">
  <a href="https://colab.research.google.com/drive/"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"></a>
  <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/C++-11+-00599C?logo=c%2B%2B" alt="C++"></a>
</p>

> A scalable, high-performance system that combines **Neural Collaborative Filtering** for product recommendations with **Logistics Optimization** for warehouse order assignment — powered by C++ cores and distributed computing.

---

## 🚀 Overview

**OmniScale** is an end-to-end machine learning system designed for two critical e-commerce tasks:

1. **Neural Recommender** — Predicts user preferences using Deep Learning (NCF model)
2. **Logistics Optimizer** — Assigns orders to warehouses to minimize delivery distance under capacity constraints

The project demonstrates a complete MLOps pipeline: from raw data parsing → feature engineering → model training → high-performance optimization → distributed execution.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OmniScale Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Phase 1:     │ →  │ Phase 2:     │ →  │ Phase 3:        │  │
│  │ Data Parsing │    │ Feature     │    │ Neural          │  │
│  │ & Cleaning   │    │ Mining       │    │ Recommender     │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         ↓                   ↓                    ↓              │
│  Stream Amazon       K-Means++           PyTorch NCF           │
│  JSON Reviews       Clustering          Model                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ Phase 4:     │ →  │ Phase 5:     │                          │
│  │ HPC          │    │ Distributed  │                          │
│  │ Optimizer    │    │ Computing    │                          │
│  └──────────────┘    └──────────────┘                          │
│         ↓                   ↓                                    │
│  C++ / OpenMP         MapReduce                                │
│  Solver               Simulation                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
OmniScale-Optimizer/
├── data/
│   ├── raw/                    # Raw Amazon reviews data
│   └── processed/              # Cleaned & feature-engineered data
├── src/
│   ├── parser/
│   │   ├── stream_parser.py   # JSON stream parser
│   │   └── feature_miner.py   # Feature extraction & K-Means++
│   ├── models/
│   │   └── ncf_model.py       # Neural Collaborative Filtering
│   ├── optimizer/
│   │   ├── solver.py          # Python logistics solver
│   │   └── cpp_core/
│   │       ├── optimizer.cpp  # C++ HPC kernel
│   │       └── fast_optimizer.so  # Compiled pybind11 module
│   └── distributed/
│       └── map_reduce_ops.py  # Distributed MapReduce simulation
├── notebooks/
│   └── OmniScale-Optimizer.ipynb  # Main execution notebook
├── scripts/
│   └── (utility scripts)
└── README.md
```

---

## ✨ Key Features

### 🧠 Neural Collaborative Filtering (NCF)
- **Deep Learning** recommender using PyTorch
- Embedding layers for users and items
- Multi-layer perceptron (MLP) for non-linear feature interactions
- MSE loss with Adam optimizer

### 📦 Logistics Optimization
- **K-Means++** clustering for delivery zone discovery
- Greedy assignment with capacity constraints
- Minimizes total distance from customers to warehouses

### ⚡ High-Performance Computing (HPC)
- **C++ core** with OpenMP parallelization
- **pybind11** bindings for Python integration
- GPU-ready architecture (CUDA support)

### 🌐 Distributed Computing
- **MapReduce** simulation using `ProcessPoolExecutor`
- Parallel order assignment across multiple workers
- Scalable to real cluster deployments (AWS EC2, GCP)

---

## 📊 Usage

### Phase 1: Data Parsing
```
python
from src.parser.stream_parser import stream_amazon_data

# Stream JSON reviews one at a time
for record in stream_amazon_data('data/raw/reviews_Electronics_5.json'):
    print(record)
```

### Phase 2: Feature Mining
```
python
from src.parser.feature_miner import FeatureMiner

miner = FeatureMiner('data/raw/reviews_Electronics_5.json')
df = miner.extract_interactions(limit=100000)

# Cluster users into 10 delivery zones
centroids, labels = miner.manual_kmeans(df[['lat', 'lon']].values, k=10)
```

### Phase 3: Train Recommender
```
python
from src.models.ncf_model import NCFModel
import torch.optim as optim

model = NCFModel(num_users=5000, num_items=10000, embed_size=32)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop...
```

### Phase 4: HPC Optimization
```
python
from src.optimizer.solver import LogisticsOptimizer

optimizer = LogisticsOptimizer(warehouse_locations, capacities)
assignments, usage = optimizer.assign_orders(user_locations)
```

### Phase 5: Distributed Execution
```
python
from src.distributed.map_reduce_ops import run_distributed_optimizer

assignments, usage = run_distributed_optimizer(
    df, warehouse_coords, caps, num_workers=4
)
```

---

## 📈 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| NCF Model | Embedding Size | 32 |
| NCF Model | Training Epochs | 5 |
| K-Means++ | Clusters (Zones) | 10 |
| C++ Optimizer | Parallelization | OpenMP |
| Distributed | Workers | 4 (simulated) |

---

## 📚 Technologies Used

- **Python 3.8+** — Core language
- **PyTorch 2.0+** — Neural network framework
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **pybind11** — Python/C++ bindings
- **C++11** — High-performance core
- **OpenMP** — Parallel computing
- **Google Colab** — Cloud execution environment

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>OmniScale</strong> — Scaling Intelligence from Recommendation to Delivery
</p>
