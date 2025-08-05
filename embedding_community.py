
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class CommunityEmbedding(nn.Module):
    def __init__(self, num_nodes, num_communities, temperature=10.0, noise_beta=1.0):
        super(CommunityEmbedding, self).__init__()
        self.p = nn.Parameter(torch.randn(num_nodes, num_communities))
        self.temperature = temperature
        self.noise_beta = noise_beta

    def forward(self, A):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.p) + 1e-10) + 1e-10)
        logits = (self.p + gumbel_noise * self.noise_beta) / self.temperature
        M = F.one_hot(torch.argmax(logits, dim=1), num_classes=logits.size(1)).float()
        A = A.float()
        A_com = M.T @ A @ M
        return M, A_com

    def community_loss(self, A_com):
        return -torch.trace(A_com)

def build_adjacency_matrix(graph: nx.Graph):
    A = nx.to_numpy_array(graph)
    return torch.tensor(A, dtype=torch.float32)
