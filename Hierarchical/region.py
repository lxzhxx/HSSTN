import math
import numpy as np
import torch
import torch.nn as nn
from radfa import RADFA
from utils import ExpMemory_lambs, Memory_lambs
from agg import get_message_aggregator
from function import get_message_function
from update import get_memory_updater
from embedding import get_embedding_module
import pandas as pd


class Region(nn.Module):
    def __init__(self, n_nodes, node_features, embedding_dimension, memory_dimension,
                 message_dimension, lambs, device, output,
                 adj_matrix_path='liuyang_adj_matrix.npy', init_lamb=0.2):
        super(Region, self).__init__()
        self._initialize_parameters(n_nodes, node_features, embedding_dimension,
                                    memory_dimension, message_dimension,
                                    lambs, device, output, init_lamb)
        self._initialize_memory_modules()
        self._initialize_message_modules()
        self._initialize_embedding_modules()
        self._initialize_attention_modules()
        self._load_adjacency_matrix(adj_matrix_path)
        self._initialize_radfa_module()

    def _initialize_parameters(self, n_nodes, node_features, embedding_dimension,
                               memory_dimension, message_dimension,
                               lambs, device, output, init_lamb):
        self.n_nodes = n_nodes
        self.node_features = node_features
        self.embedding_dimension = embedding_dimension
        self.output = output
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.lambs = torch.Tensor(lambs).to(self.device) * self.output
        self.lamb_len = self.lambs.shape[0]
        self.n_regions = math.ceil(math.sqrt(self.n_nodes))
        self.n_graphs = 4
        self.n_heads = 8
        self.lamb = nn.Parameter(torch.Tensor([init_lamb]), requires_grad=True)

        # Calculate derived dimensions
        self.message_per_head = message_dimension // self.n_heads
        self.features_per_head = self.memory_dimension // self.n_heads
        self.message_multi_head = self.message_per_head * self.n_heads
        self.features_multi_head = self.features_per_head * self.n_heads
        self.raw_message_dimension = self.memory_dimension * self.lamb_len + self.node_features.shape[1]

    def _initialize_memory_modules(self):
        self.memory = ExpMemory_lambs(
            n_nodes=self.n_nodes,
            memory_dimension=self.memory_dimension,
            lambs=self.lambs,
            device=self.device
        )
        self.region_memory = Memory_lambs(
            n_nodes=self.n_regions,
            memory_dimension=self.memory_dimension,
            lambs=self.lambs,
            device=self.device
        )
        self.graph_memory = Memory_lambs(
            n_nodes=self.n_graphs,
            memory_dimension=self.memory_dimension,
            lambs=self.lambs,
            device=self.device
        )

    def _initialize_message_modules(self):
        self.message_aggregator = get_message_aggregator(
            aggregator_type="exp_lambs",
            device=self.device,
            embedding_dimension=self.memory_dimension
        )
        self.message_function = get_message_function(
            module_type="mlp",
            raw_message_dimension=self.raw_message_dimension,
            message_dimension=self.message_dimension
        )

    def _initialize_memory_updaters(self):
        self.memory_updater = get_memory_updater(
            module_type="exp_lambs",
            memory=self.memory,
            message_dimension=self.message_dimension,
            memory_dimension=self.lambs,
            device=self.device
        )
        self.region_memory_updater = get_memory_updater(
            "exp_lambs",
            self.region_memory,
            self.message_dimension,
            self.lambs,
            self.device
        )
        self.graph_memory_updater = get_memory_updater(
            "exp_lambs",
            self.graph_memory,
            self.message_dimension,
            self.lambs,
            self.device
        )

    def _initialize_embedding_modules(self):
        self.exp_embedding = get_embedding_module(module_type="exp_lambs")
        self.iden_embedding = get_embedding_module(module_type="identity")
        self.static_embedding = nn.Embedding(self.n_nodes, self.embedding_dimension)

    def _initialize_attention_modules(self):
        self.Q_r = nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.K_r = nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.V_r = nn.Linear(self.raw_message_dimension, self.message_multi_head, bias=False)

        self.Q_g = nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.K_g = nn.Linear(self.memory_dimension * self.lamb_len, self.features_multi_head, bias=False)
        self.V_g = nn.Linear(self.message_multi_head, self.message_multi_head, bias=False)

        self.fution1 = nn.Linear(256, 1)
        self.fution2 = nn.Linear(256, 1)
        self.softmax_fution = nn.Softmax(dim=-1)
        self.ff_r = nn.Sequential(
            nn.Linear(self.message_multi_head, self.message_multi_head, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.message_multi_head, self.message_dimension, bias=True),
            nn.LeakyReLU()
        )
        self.ff_g = nn.Sequential(
            nn.Linear(self.message_multi_head, self.message_multi_head, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.message_multi_head, self.message_dimension, bias=True),
            nn.LeakyReLU()
        )
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.memory_dimension * self.lamb_len, self.memory_dimension, bias=True),
            nn.LeakyReLU()
        )
        self.spatial_transform = nn.Sequential(
            nn.Linear(self.memory_dimension * 3, self.embedding_dimension, bias=True),
            nn.LeakyReLU()
        )

    def _load_adjacency_matrix(self, adj_matrix_path):
        adj_matrix = np.load(adj_matrix_path)
        self.adj_matrix = torch.from_numpy(adj_matrix).float().to(self.device)

    def _initialize_radfa_module(self):
        self.radfa = RADFA(
            dim=256,
            qk_dim=64,
            mlp_dim=128,
            adj_matrix=self.adj_matrix,
            heads=1,
            topk=3
        ).to(self.device)

    def forward(self, source_nodes, target_nodes, edge_timediff, timestamps_batch_torch,
                now_time, begin_time, predict_od):
        # Get initial memory and compute target embeddings
        memory = self.memory.get_memory()
        target_embeddings = self.exp_embedding.compute_embedding(
            memory=memory,
            nodes=target_nodes,
            time_diffs=edge_timediff
        )

        # Process raw messages
        raw_messages = self._get_raw_messages(
            source_nodes,
            target_embeddings,
            edge_timediff,
            self.node_features[target_nodes],
            timestamps_batch_torch
        )
        unique_nodes, unique_raw_messages, unique_timestamps = self.message_aggregator.aggregate(
            source_nodes,
            raw_messages,
            self.lambs
        )
        unique_messages = torch.cat([
            self.message_function.compute_message(unique_raw_messages[:, :, :-1]),
            unique_raw_messages[:, :, -1:]
        ], dim=-1)

        # Get region and graph memory states
        region_memory = self.region_memory.get_memory()
        graph_memory = self.graph_memory.get_memory()
        last_update = self.memory.last_update
        time_diffs = -last_update + begin_time
        static_node_embedding = self._compute_static_node_embeddings(memory, time_diffs)
        region_embedding, graph_embedding = self._compute_region_graph_embeddings(
            region_memory,
            graph_memory,
            time_diffs
        )
        A_r, A_g = self._compute_attention_weights(region_embedding, static_node_embedding, graph_embedding)
        region_messages, graph_messages = self._compute_attention_messages(
            A_r,
            A_g,
            unique_nodes,
            unique_raw_messages,
            region_embedding
        )
        updated_memory, updated_last_update = self._update_memories(
            unique_nodes,
            unique_messages,
            unique_timestamps,
            region_messages,
            graph_messages,
            now_time
        )
        embeddings = self._compute_final_embeddings(
            updated_memory,
            updated_last_update,
            now_time,
            A_r,
            A_g
        )

        return embeddings

    def _get_raw_messages(self, source_nodes, target_embeddings, edge_timediff, node_features, edge_times):
        source_message = torch.cat([
            target_embeddings,
            node_features,
            torch.ones([target_embeddings.shape[0], 1]).to(self.device)
        ], dim=1)
        messages = {
            node: [source_message[source_nodes == node], edge_times[source_nodes == node]]
            for node in np.unique(source_nodes)
        }
        return messages

    def _compute_static_node_embeddings(self, memory, time_diffs):
        static_node_embedding = self.exp_embedding.compute_embedding(
            memory=memory,
            nodes=list(range(self.n_nodes)),
            time_diffs=time_diffs
        )
        data_tensor = self._load_csv_data('Data/liuyang.csv')
        features = self._extract_features(data_tensor).to(self.device)

        alpha1 = self.fution1(static_node_embedding).squeeze(-1)
        alpha2 = self.fution2(features).squeeze(-1)
        attention_weights = self.softmax_fution(torch.stack([alpha1, alpha2], dim=-1))

        return (attention_weights[..., 0].unsqueeze(-1) * static_node_embedding + \
                (attention_weights[..., 1].unsqueeze(-1) * features))

    def _compute_region_graph_embeddings(self, region_memory, graph_memory, time_diffs):
        region_embedding = self.iden_embedding.compute_embedding(
            memory=region_memory,
            nodes=list(range(self.n_regions)),
            time_diffs=time_diffs
        ).reshape([self.n_regions, -1])

        seq_len, feature_dim = region_embedding.size(0), region_embedding.size(1)
        x = region_embedding.view(1, seq_len, feature_dim)
        region_indices = self._load_region_indices('Data/liuyang_indices.csv')
        output = self.radfa(x, region_indices)
        region_embedding = output.view(seq_len, feature_dim)

        graph_embedding = self.iden_embedding.compute_embedding(
            memory=graph_memory,
            nodes=list(range(self.n_graphs)),
            time_diffs=time_diffs
        ).reshape([self.n_graphs, -1])

        return region_embedding, graph_embedding

    def _compute_attention_weights(self, region_embedding, static_node_embedding, graph_embedding):
        A_r = torch.einsum(
            "rhf,nhf->rhn",
            self.Q_r(region_embedding).reshape([self.n_regions, self.n_heads, self.features_per_head]),
            self.K_r(static_node_embedding).reshape([self.n_nodes, self.n_heads, self.features_per_head])
        ) / math.sqrt(self.features_per_head)

        A_g = torch.einsum(
            "ghf,rhf->ghr",
            self.Q_g(graph_embedding).reshape([self.n_graphs, self.n_heads, self.features_per_head]),
            self.K_g(region_embedding).reshape([self.n_regions, self.n_heads, self.features_per_head])
        ) / math.sqrt(self.features_per_head)

        return A_r, A_g

    def _compute_attention_messages(self, A_r, A_g, unique_nodes, unique_raw_messages, region_embedding):
        region_messages_mid = torch.einsum(
            "rhn,nlhf->rlhf",
            torch.softmax(A_r[:, :, unique_nodes], dim=2),
            self.V_r((unique_raw_messages[:, :, :-1] / unique_raw_messages[:, :, -1:])
                     .reshape(len(unique_nodes), self.lamb_len, -1))
                .reshape([len(unique_nodes), self.lamb_len, self.n_heads, self.message_per_head])
        ).reshape([self.n_regions, self.lamb_len, self.message_multi_head])

        region_messages = self.ff_r(region_messages_mid)

        graph_messages_mid = torch.einsum(
            "ghr,rlhf->glhf",
            torch.softmax(A_g, dim=2),
            self.V_g(region_messages_mid).reshape(
                [self.n_regions, self.lamb_len, self.n_heads, self.message_per_head])
        )
        graph_messages = self.ff_g(graph_messages_mid.reshape(
            [self.n_graphs, self.lamb_len, self.message_multi_head]))

        return region_messages, graph_messages

    def _update_memories(self, unique_nodes, unique_messages, unique_timestamps,
                         region_messages, graph_messages, now_time):
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
            unique_nodes,
            unique_messages,
            timestamps=unique_timestamps
        )

        self.region_memory_updater.update_memory(
            list(range(self.n_regions)),
            region_messages,
            timestamps=now_time
        )
        self.graph_memory_updater.update_memory(
            list(range(self.n_graphs)),
            graph_messages,
            timestamps=now_time
        )

        return updated_memory, updated_last_update

    def _compute_final_embeddings(self, updated_memory, updated_last_update, now_time, A_r, A_g):
        updated_time_diffs = -updated_last_update + now_time

        recent_node_embeddings = self.exp_embedding.compute_embedding(
            memory=updated_memory,
            nodes=list(range(self.n_nodes)),
            time_diffs=updated_time_diffs
        )

        recent_region_embeddings = self.iden_embedding.compute_embedding(
            memory=self.region_memory.get_memory(),
            nodes=list(range(self.n_regions))
        ).reshape([self.n_regions, -1])

        recent_graph_embeddings = self.iden_embedding.compute_embedding(
            memory=self.graph_memory.get_memory(),
            nodes=list(range(self.n_graphs))
        ).reshape([self.n_graphs, -1])

        r2n = torch.mean(torch.softmax(A_r, dim=0), dim=1)
        g2r = torch.mean(torch.softmax(A_g, dim=0), dim=1)

        region_node_embeddings = torch.mm(r2n.T, recent_region_embeddings)
        graph_node_embeddings = torch.mm(torch.mm(r2n.T, g2r.T), recent_graph_embeddings)

        dynamic_embeddings = torch.cat([
            recent_node_embeddings,
            region_node_embeddings,
            graph_node_embeddings
        ], dim=0)

        embeddings = self.lamb * self.static_embedding.weight + \
                     (1 - self.lamb) * self.spatial_transform(
            self.embedding_transform(dynamic_embeddings)
                .reshape([3, self.n_nodes, self.memory_dimension])
                .permute([1, 0, 2])
                .reshape([self.n_nodes, -1]))

        return embeddings

    def _load_csv_data(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return torch.tensor(data.values, dtype=torch.float).to(self.device)

    def _extract_features(self, data_tensor):
        model = FeatureExtractor().to(self.device)
        features = model(data_tensor)
        return features

    def _load_region_indices(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        region_indices = df[0].astype(int).tolist()
        return region_indices

    def init_memory(self):
        self.memory.__init_memory__()
        self.region_memory.__init_memory__()
        self.graph_memory.__init_memory__()

    def backup_memory(self):
        return [
            self.memory.backup_memory(),
            self.region_memory.backup_memory(),
            self.graph_memory.backup_memory()
        ]

    def restore_memory(self, memory):
        self.memory.restore_memory(memory[0])
        self.region_memory.restore_memory(memory[1])
        self.graph_memory.restore_memory(memory[2])

    def detach_memory(self):
        self.memory.detach_memory()
        self.region_memory.detach_memory()
        self.graph_memory.detach_memory()