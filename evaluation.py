
import torch

def topk_accuracy(alignment_matrix: torch.Tensor, anchor_links: list, topk: list = [1, 5, 10, 30]) -> dict:
    """
    Compute TOP-K accuracy for alignment.
    alignment_matrix: (ns, nt) tensor with alignment scores
    anchor_links: list of (source_node_idx, target_node_idx)
    topk: list of K values to evaluate
    Returns:
        Dictionary of {k: accuracy}
    """
    result = {}
    alignment_matrix = alignment_matrix.detach().cpu()
    for k in topk:
        hits = 0
        for (src_idx, tgt_idx) in anchor_links:
            if src_idx >= alignment_matrix.shape[0] or tgt_idx >= alignment_matrix.shape[1]:
                continue
            row = alignment_matrix[src_idx]
            topk_indices = torch.topk(row, k=k).indices
            if tgt_idx in topk_indices:
                hits += 1
        result[f"TOP@{k}"] = hits / len(anchor_links) if anchor_links else 0.0
    return result
