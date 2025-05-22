import numpy as np
import torch
from torch import nn
from data import Data

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance

def to_device(data, device):
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    elif isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.Tensor(data).to(device)

class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, nodes, time_diffs=None):
        return memory[nodes, :]


class ExpLambsEmbedding(EmbeddingModule):
    def __init__(self):
        super(ExpLambsEmbedding, self).__init__()

    def compute_embedding(self, memory, nodes, time_diffs=None):
        embeddings = (memory[nodes, :, :-1] / memory[nodes, :, -1:]).reshape([len(nodes), -1])
        return embeddings

class Memory_lambs(nn.Module):

    def __init__(self, n_nodes, memory_dimension, lambs, device="cpu"):
        super(Memory_lambs, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.lambs = lambs
        self.lamb_len = lambs.shape[0]
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.lamb_len, self.memory_dimension)),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()

class ExpMemory_lambs(nn.Module):
    def __init__(self, n_nodes, memory_dimension, lambs, device="cpu"):
        super(ExpMemory_lambs, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.lambs = lambs
        self.lamb_len = lambs.shape[0]
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(torch.cat([torch.zeros((self.n_nodes, self.lamb_len, self.memory_dimension)),
                                              torch.ones((self.n_nodes, self.lamb_len, 1))], dim=2),
                                   requires_grad=False).to(self.device)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False).to(self.device)

    def get_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    def detach_memory(self):
        self.memory.detach_()

def get_od_data(config):
    whole_data = np.load(config["data_path"]).astype("int").reshape([-1, 3])
    od_matrix = np.load(config["matrix_path"])
    back_points = np.load(config["point_path"])
    print("data loaded")
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"] - config["input_len"], (config["train_day"] + config["val_day"]) * config["day_cycle"] - config["input_len"]
    origin = whole_data[:, 0]
    destinations = whole_data[:, 1]
    timestamps = whole_data[:, 2]
    edge_idxs = np.arange(whole_data.shape[0])
    n_nodes = config["n_nodes"]
    node_features = np.diag(np.ones(n_nodes))
    full_data = Data(origin, destinations, timestamps, edge_idxs, n_nodes)

    return n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points

