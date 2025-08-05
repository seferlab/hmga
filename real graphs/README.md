
# HMGA: Hyperbolic Graph Learning for Multi-Perspective Graph Alignment

This project implements the HMGA framework for aligning biological or synthetic graphs using first-order, hyperbolic, and higher-order structural information.

## 📁 Modules

- `synthetic_data.py`: NAPAbench-style synthetic graph and anchor generator.
- `load_real_graphs.py`: Loads real-world graphs and anchor links from files.
- `embedding_first_order.py`: First-order embeddings with skip-gram negative sampling + anchor mapping loss.
- `embedding_hyperbolic.py`: Hyperbolic–Euclidean GCN model with Möbius interaction and training.
- `embedding_community.py`: Community-aware embeddings using Gumbel-softmax and self-supervised training.
- `alignment.py`: Alignment matrix computation and fusion.
- `evaluation.py`: TOP@K accuracy evaluation.
- `run_pipeline.py`: End-to-end alignment pipeline.

## ▶️ Usage

### Run with Synthetic Data

```bash
python run_pipeline.py
```

### Run with Real-World Graphs

```bash
python run_pipeline.py --real \
  --source_graph data/source.edgelist \
  --target_graph data/target.edgelist \
  --anchor_links data/anchor_links.csv \
  --format edgelist
```

Supported formats: `edgelist`, `csv`, `gml`, `graphml`

## ⚙️ Requirements

- Python 3.7+
- PyTorch
- NetworkX
- NumPy
- pandas

Install all dependencies:

```bash
pip install torch networkx numpy pandas
```

## 🧠 Model Components

- **First-order embedding**: Random walk-based skip-gram training with negative sampling
- **Hyperbolic–Euclidean GCN**: Multi-perspective message passing using Möbius operations
- **Community embedding**: Gumbel-softmax community detection and adjacency compression
- **Fusion & Evaluation**: Combined similarity matrices are evaluated using TOP-K accuracy

## 🧪 Notes

- Training epochs and hyperparameters can be modified in `run_pipeline.py`
- Default settings use 128-dimensional embeddings and 20 communities

## 📄 License

MIT License
