
import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclideanGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EuclideanGATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_dim))

    def forward(self, h, adj):
        Wh = self.W(h)
        N = h.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, N, 2 * Wh.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class HyperbolicUtils:
    @staticmethod
    def mobius_add(x, y, k=1.0):
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        y2 = (y ** 2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * k * xy + k * y2) * x + (1 - k * x2) * y
        denom = 1 + 2 * k * xy + k ** 2 * x2 * y2
        return num / (denom + 1e-5)

    @staticmethod
    def mobius_matvec(m, x, k=1.0):
        mx = x @ m.transpose(-1, -2)
        return HyperbolicUtils.exp_map_zero(HyperbolicUtils.log_map_zero(mx, k), k)

    @staticmethod
    def exp_map_zero(x, k=1.0):
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-5)
        return torch.tanh(torch.sqrt(torch.tensor(k)) * norm) * x / (torch.sqrt(torch.tensor(k)) * norm)

    @staticmethod
    def log_map_zero(x, k=1.0):
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-5)
        return torch.atanh(torch.sqrt(torch.tensor(k)) * norm) * x / (torch.sqrt(torch.tensor(k)) * norm)

    @staticmethod
    def hyperbolic_distance(x, y, k=1.0):
        diff = HyperbolicUtils.mobius_add(-x, y, k)
        norm = torch.clamp(torch.norm(diff, dim=-1), min=1e-5)
        return (2 / torch.sqrt(torch.tensor(k))) * torch.atanh(torch.sqrt(torch.tensor(k)) * norm)


class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, k=1.0):
        super(HyperbolicLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.k = k

    def forward(self, x):
        log_x = HyperbolicUtils.log_map_zero(x, self.k)
        out = self.linear(log_x)
        return HyperbolicUtils.exp_map_zero(out, self.k)


class InteractiveHyperbolicGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, k=1.0):
        super(InteractiveHyperbolicGCN, self).__init__()
        self.gat_euc = EuclideanGATLayer(in_dim, hidden_dim)
        self.hyp_linear = HyperbolicLinear(in_dim, hidden_dim, k)
        self.k = k
        self.mu = nn.Parameter(torch.tensor(0.5))
        self.mu_prime = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, adj):
        h_euc = self.gat_euc(x, adj)
        h_hyp = self.hyp_linear(HyperbolicUtils.exp_map_zero(x, self.k))

        h_hyp_updated = HyperbolicUtils.mobius_add(
            h_hyp, self.mu * HyperbolicUtils.exp_map_zero(h_euc, self.k), self.k
        )
        h_euc_updated = h_euc + self.mu_prime * HyperbolicUtils.log_map_zero(h_hyp, self.k)

        return h_euc_updated, h_hyp_updated


def train_hyperbolic_model(model, features, adj, epochs=200, lr=0.01, device="cpu"):
    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        Z_euc, Z_hyp = model(features, adj)
        # Dummy self-supervised loss (e.g., preserve norm structure)
        loss = torch.mean(torch.norm(Z_hyp, dim=1)) + torch.mean(torch.norm(Z_euc, dim=1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] HyperbolicGCN loss: {loss.item():.4f}")
