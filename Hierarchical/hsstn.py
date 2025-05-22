import torch
import torch.nn as nn
import numpy as np
from .region import Region
from .prediction import PredictionLayer


class HSSTNet(nn.Module):
    def __init__(self, device,
                 n_nodes=279, node_features=None,
                 message_dimension=256, memory_dimension=256, lambs=None,
                 output=30):
        super(HSSTNet, self).__init__()
        if lambs is None:
            lambs = [1]
        node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        embedding_dimension = memory_dimension * 1
        self.memory = Region(n_nodes, node_raw_features, embedding_dimension, memory_dimension,
                                 message_dimension, lambs, device,
                                 output, init_lamb=0.3)
        self.predict_od = PredictionLayer(embedding_dimension, n_nodes)

    def compute_od_matrix(self, o_nodes, d_nodes, timestamps_batch_torch,
                          edge_timediff, now_time, begin_time,
                          predict_od=True):
        embeddings = self.memory(o_nodes, d_nodes, edge_timediff,
                                       timestamps_batch_torch, now_time, begin_time,
                                       predict_od)
        od_matrix = None
        if predict_od:
            od_matrix = self.predict_od(embeddings)

        return od_matrix

    def init_memory(self):
        self.memory.init_memory()

    def backup_memory(self):
        return self.memory.backup_memory()

    def restore_memory(self, memories):
        self.memory.restore_memory(memories)

    def detach_memory(self):
        self.memory.detach_memory()