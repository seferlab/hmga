
import argparse
import torch
import networkx as nx
from synthetic_data import generate_napabench_like_graphs
from load_real_graphs import load_real_graphs
from embedding_first_order import FirstOrderEmbeddingModel, AnchorMappingLoss, train_first_order
from embedding_hyperbolic import InteractiveHyperbolicGCN, train_hyperbolic_model
from embedding_community import CommunityEmbedding, build_adjacency_matrix, train_community_embedding
from alignment import compute_alignment_matrix, fuse_alignment_matrices
from evaluation import topk_accuracy

def run_pipeline(use_real_data=False, real_data_paths=None, device="cpu"):
    # 1. Load Graphs
    if use_real_data and real_data_paths:
        Gs, Gt, anchor_links = load_real_graphs(
            real_data_paths["source_graph"],
            real_data_paths["target_graph"],
            real_data_paths["anchor_links"],
            fmt=real_data_paths.get("format", "edgelist")
        )
    else:
        _, Gs, Gt, anchor_links = generate_napabench_like_graphs()

    Ns, Nt = Gs.number_of_nodes(), Gt.number_of_nodes()
    print(f"Source: {Ns} nodes, Target: {Nt} nodes, Anchors: {len(anchor_links)}")

    # 2. First-order embeddings with training
    model_s = FirstOrderEmbeddingModel(Ns, 128)
    model_t = FirstOrderEmbeddingModel(Nt, 128)
    train_first_order(model_s, Gs, epochs=50, lr=0.01, device=device)
    train_first_order(model_t, Gt, epochs=50, lr=0.01, device=device)

    anchor_loss = AnchorMappingLoss(embedding_dim=128).to(device)
    Zs_first = model_s.get_embeddings().to(device)
    Zt_first = model_t.get_embeddings().to(device)
    loss = anchor_loss(Zs_first, Zt_first, anchor_links)
    Zs_mapped = anchor_loss.map(Zs_first)
    S1 = compute_alignment_matrix(Zs_mapped, Zt_first)

    # 3. Hyperbolic-Euclidean embeddings with training
    x_s = torch.randn(Ns, 128).to(device)
    x_t = torch.randn(Nt, 128).to(device)
    A_s = build_adjacency_matrix(Gs).to(device)
    A_t = build_adjacency_matrix(Gt).to(device)

    hyp_model = InteractiveHyperbolicGCN(128, 128).to(device)
    train_hyperbolic_model(hyp_model, x_s, A_s, epochs=50, lr=0.01, device=device)
    Zs_euc, Zs_hyp = hyp_model(x_s, A_s)
    Zt_euc, Zt_hyp = hyp_model(x_t, A_t)
    S2 = compute_alignment_matrix(Zs_hyp, Zt_hyp)
    S3 = compute_alignment_matrix(Zs_euc, Zt_euc)

    # 4. Community-aware embedding
    com_s = CommunityEmbedding(Ns, num_communities=20)
    com_t = CommunityEmbedding(Nt, num_communities=20)
    train_community_embedding(com_s, A_s, epochs=30, lr=0.01, device=device)
    train_community_embedding(com_t, A_t, epochs=30, lr=0.01, device=device)
    _, Acom_s = com_s(A_s)
    _, Acom_t = com_t(A_t)
    S4 = compute_alignment_matrix(Acom_s, Acom_t)

    # 5. Fuse matrices and evaluate
    S_final = fuse_alignment_matrices([S1, S2, S3, S4])
    results = topk_accuracy(S_final, anchor_links)
    print("Alignment Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real-world dataset")
    parser.add_argument("--source_graph", type=str, default="")
    parser.add_argument("--target_graph", type=str, default="")
    parser.add_argument("--anchor_links", type=str, default="")
    parser.add_argument("--format", type=str, default="edgelist")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.real:
        real_data_paths = {
            "source_graph": args.source_graph,
            "target_graph": args.target_graph,
            "anchor_links": args.anchor_links,
            "format": args.format
        }
        run_pipeline(use_real_data=True, real_data_paths=real_data_paths, device=args.device)
    else:
        run_pipeline(use_real_data=False, real_data_paths=None, device=args.device)
