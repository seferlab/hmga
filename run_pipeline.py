
import torch
import networkx as nx
from synthetic_data import generate_napabench_like_graphs
from embedding_first_order import FirstOrderEmbeddingModel, AnchorMappingLoss
from embedding_hyperbolic import InteractiveHyperbolicGCN
from embedding_community import CommunityEmbedding, build_adjacency_matrix
from alignment import compute_alignment_matrix, fuse_alignment_matrices
from evaluation import topk_accuracy

def run_pipeline():
    # 1. Generate synthetic graphs
    ancestor, Gs, Gt, anchor_links = generate_napabench_like_graphs()
    Ns, Nt = Gs.number_of_nodes(), Gt.number_of_nodes()
    print(f"Source: {Ns} nodes, Target: {Nt} nodes, Anchors: {len(anchor_links)}")

    # 2. First-order embeddings
    first_order_model_s = FirstOrderEmbeddingModel(Ns, 128)
    first_order_model_t = FirstOrderEmbeddingModel(Nt, 128)
    anchor_loss = AnchorMappingLoss(embedding_dim=128)

    # Simulated embeddings (training skipped)
    Zs_first = first_order_model_s.get_embeddings()
    Zt_first = first_order_model_t.get_embeddings()
    Zs_first_mapped = anchor_loss.map(Zs_first)

    S1 = compute_alignment_matrix(Zs_first_mapped, Zt_first)

    # 3. Hyperbolic-Euclidean GCN embeddings (simulated)
    x_s = torch.randn(Ns, 128)
    x_t = torch.randn(Nt, 128)
    A_s = build_adjacency_matrix(Gs)
    A_t = build_adjacency_matrix(Gt)

    hyp_model = InteractiveHyperbolicGCN(128, 128)
    Zs_euc, Zs_hyp = hyp_model(x_s, A_s)
    Zt_euc, Zt_hyp = hyp_model(x_t, A_t)

    S2 = compute_alignment_matrix(Zs_hyp, Zt_hyp)
    S3 = compute_alignment_matrix(Zs_euc, Zt_euc)

    # 4. Community-aware embedding
    com_model_s = CommunityEmbedding(Ns, num_communities=20)
    com_model_t = CommunityEmbedding(Nt, num_communities=20)
    _, Acom_s = com_model_s(A_s)
    _, Acom_t = com_model_t(A_t)

    S4 = compute_alignment_matrix(Acom_s, Acom_t)

    # 5. Fuse all matrices
    S_final = fuse_alignment_matrices([S1, S2, S3, S4])

    # 6. Evaluate
    results = topk_accuracy(S_final, anchor_links)
    print("Alignment Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_pipeline()
