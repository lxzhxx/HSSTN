import torch
import torch.nn as nn

class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes):
        super(PredictionLayer, self).__init__()
        self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, embeddings):
        return self.w(torch.cat(
            [embeddings.repeat([1, self.n_nodes]).reshape([self.n_nodes * self.n_nodes, -1]),
             embeddings.repeat([self.n_nodes, 1])],
            dim=1)).reshape([self.n_nodes, self.n_nodes])