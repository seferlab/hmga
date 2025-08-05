
# HMGA: Hyperbolic Graph Learning for Multi-Perspective Graph Alignment

This project implements the HMGA framework for aligning biological or synthetic graphs using first-order, hyperbolic, and higher-order structural information.

## 📁 Files

- `synthetic_data.py`: Generates NAPAbench-like synthetic graphs and anchor links.
- `embedding_first_order.py`: First-order structural embedding and anchor mapping.
- `embedding_hyperbolic.py`: Hyperbolic-Euclidean GCN with interactive updates.
- `embedding_community.py`: Gumbel-softmax based community-aware embeddings.
- `alignment.py`: Alignment matrix computation and fusion.
- `evaluation.py`: TOP@K accuracy evaluation.
- `run_pipeline.py`: Main script to run the full graph alignment pipeline.

## ▶️ Usage

To run the full HMGA pipeline:

```bash
python run_pipeline.py
```

This will:
1. Generate synthetic source and target graphs
2. Compute embeddings using multiple methods
3. Fuse alignment matrices
4. Evaluate using TOP-1, TOP-5, TOP-10, and TOP-30 accuracy

## ⚙️ Requirements

- Python 3.7+
- PyTorch
- NetworkX
- NumPy

Install dependencies using:

```bash
pip install torch networkx numpy
```

## 🧪 Notes

- This version uses synthetic data only (from NAPAbench-like generation).
- Embedding models are randomly initialized; full training routines can be added later.

## 📄 License

MIT License
