
import torch
import torch.nn as nn
import torch.nn.functional as F

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
