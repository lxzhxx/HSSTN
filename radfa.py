import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

class RADFA(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, adj_matrix=None, region_memory=None, heads=8, topk=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (qk_dim // heads) ** -0.5
        self.topk = topk

        self.geo_neighbors = self._process_adj_matrix(adj_matrix) if adj_matrix is not None else None

        if region_memory is not None:
            self.register_buffer('region_memory', region_memory)
            self.cluster_neighbors = self._find_cluster_neighbors(region_memory, topk)
        else:
            self.cluster_neighbors = None

        self.to_q = nn.Linear(dim, qk_dim)
        self.to_k = nn.Linear(dim, qk_dim)
        self.to_v = nn.Linear(dim, qk_dim)
        self.to_out = nn.Sequential(
            nn.Linear(qk_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_dim, dropout)

        self.dynamic_gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )

    def _process_adj_matrix(self, adj_matrix):
        return [torch.nonzero(row, as_tuple=True)[0].tolist() for row in adj_matrix]

    def _find_cluster_neighbors(self, region_memory, topk):
        norms = torch.norm(region_memory, p=2, dim=1, keepdim=True)
        normalized_memory = region_memory / (norms + 1e-8)
        sim_matrix = torch.mm(normalized_memory, normalized_memory.t())
        sim_matrix.fill_diagonal_(-float('inf'))
        _, indices = torch.topk(sim_matrix, k=topk, dim=1)
        return [row.tolist() for row in indices]

    def get_extended_neighborhood(self, region_idx):
        geo_nbrs = self.geo_neighbors[region_idx] if self.geo_neighbors else []
        cluster_nbrs = self.cluster_neighbors[region_idx] if self.cluster_neighbors else []
        return list(set(geo_nbrs + cluster_nbrs))

    def forward(self, x, region_indices=None):
        batch, seq_len, _ = x.shape

        x_norm = self.attn_norm(x)

        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn_scores = torch.zeros(batch, self.heads, seq_len, seq_len, device=x.device)

        if region_indices is not None:
            for i in range(seq_len):
                src_region = region_indices[i]
                neighbors = self.get_extended_neighborhood(src_region)

                for j in neighbors:
                    if j >= seq_len:
                        continue
                    dots = einsum('b h d, b h d -> b h', q[:, :, i], k[:, :, j]) * self.scale
                    attn_scores[:, :, i, j] = dots
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn_scores = dots

        attn_weights = F.softmax(attn_scores, dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        attn_out = self.to_out(out)

        fusion_gate = self.dynamic_gate(torch.cat([x, attn_out], dim=-1))
        fused_output = fusion_gate * x + (1 - fusion_gate) * attn_out

        ffn_out = self.ffn(self.ffn_norm(fused_output))
        final_output = fused_output + ffn_out

        return final_output


class FFN(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
