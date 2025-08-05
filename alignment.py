
import torch

def compute_alignment_matrix(Zs: torch.Tensor, Zt: torch.Tensor) -> torch.Tensor:
    """
    Compute alignment matrix between source and target embeddings.
    Zs: (ns, d) source node embeddings
    Zt: (nt, d) target node embeddings
    Returns:
        (ns, nt) alignment score matrix
    """
    return Zs @ Zt.T


def fuse_alignment_matrices(matrices: list, weights: list = None) -> torch.Tensor:
    """
    Fuse multiple alignment matrices with weights.
    matrices: list of (ns, nt) tensors
    weights: list of floats, same length as matrices
    Returns:
        (ns, nt) final fused matrix
    """
    if weights is None:
        weights = [1.0 / len(matrices)] * len(matrices)
    fused = sum(w * m for w, m in zip(weights, matrices))
    return fused
