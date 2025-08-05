
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FirstOrderEmbeddingModel(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super(FirstOrderEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, node_pairs: torch.Tensor, neg_samples: torch.Tensor):
        v_i = self.embeddings(node_pairs[:, 0])
        v_j = self.embeddings(node_pairs[:, 1])
        score_pos = torch.sum(v_i * v_j, dim=1)
        loss_pos = F.logsigmoid(score_pos).mean()

        v_k = self.embeddings(neg_samples)
        v_i_exp = v_i.unsqueeze(1)
        score_neg = torch.sum(v_i_exp * v_k, dim=2)
        loss_neg = F.logsigmoid(-score_neg).mean()

        return - (loss_pos + loss_neg)

    def get_embeddings(self):
        return self.embeddings.weight


class AnchorMappingLoss(nn.Module):
    def __init__(self, mode='linear', embedding_dim=128):
        super(AnchorMappingLoss, self).__init__()
        if mode == 'linear':
            self.mapping = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif mode == 'mlp':
            self.mapping = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            raise ValueError("Unsupported mapping mode")

    def forward(self, source_embeddings, target_embeddings, anchor_links):
        src_idx = torch.tensor([a[0] for a in anchor_links], dtype=torch.long)
        tgt_idx = torch.tensor([a[1] for a in anchor_links], dtype=torch.long)
        src_embed = source_embeddings[src_idx]
        tgt_embed = target_embeddings[tgt_idx]
        mapped_src = self.mapping(src_embed)
        loss = F.mse_loss(mapped_src, tgt_embed)
        return loss

    def map(self, source_embeddings):
        return self.mapping(source_embeddings)


def generate_edge_pairs(graph, num_negatives=5):
    edges = list(graph.edges())
    node_pairs = torch.tensor(edges, dtype=torch.long)

    all_nodes = list(graph.nodes())
    neg_samples = []
    for src, _ in edges:
        negs = []
        while len(negs) < num_negatives:
            neg = random.choice(all_nodes)
            if not graph.has_edge(src, neg):
                negs.append(neg)
        neg_samples.append(negs)

    neg_samples = torch.tensor(neg_samples, dtype=torch.long)
    return node_pairs, neg_samples


def train_first_order(model, graph, epochs=100, lr=0.01, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        node_pairs, neg_samples = generate_edge_pairs(graph)
        node_pairs, neg_samples = node_pairs.to(device), neg_samples.to(device)

        optimizer.zero_grad()
        loss = model(node_pairs, neg_samples)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] First-order loss: {loss.item():.4f}")
