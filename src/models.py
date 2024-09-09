"""
Файл с описанием моделей графовых нейронных сетей
"""
import numpy as np
import torch  # type: ignore
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import BatchNorm1d, Linear, Module
from torch_geometric.nn import (ASAPooling, GATConv, GCNConv, SAGPooling,
                                TAGConv)
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool


class SkipGCN(Module):
    # The Class is derived from GCN
    def __init__(self, model_params: DictConfig, num_node_features: int):
        """
        Инициализирует экземпляр класса SkipGCN.

        Parameters
        ----------
        model_params : DictConfig
            Словарь  с конфигурацией модели, содержащий параметры hidden_channels и dropout.
        num_node_features : int
            Количество признаков в узлах графа.

        Returns
        -------
        None
        """
        hidden_channels = model_params['hidden_channels']
        # Default values for embeddings
        self.embeddings = np.ones(2 * hidden_channels + num_node_features)
        self.drop_p = model_params['dropout']
        super(SkipGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features,
                             hidden_channels,
                             #cached=True,
                             normalize=False)
        # skip connection: hidden_channels + num_node_features for residuals
        self.conv2 = GCNConv(hidden_channels + num_node_features,
                             hidden_channels + num_node_features,
                             #cached=True,
                             normalize=False)
        # skip connection: hidden_channels + hidden_channels + residuals
        self.conv3 = GCNConv(2 * hidden_channels + num_node_features,
                             2 * hidden_channels + num_node_features,
                             #cached=True,
                             normalize=False)
        self.bn_conv1 = BatchNorm1d(num_features=hidden_channels)
        self.bn_conv2 = BatchNorm1d(num_features=hidden_channels + num_node_features)
        self.bn_conv3 = BatchNorm1d(num_features=2 * hidden_channels + num_node_features)
        self.bn1 = BatchNorm1d(num_features=2 * hidden_channels + num_node_features)
        self.lin = Linear(in_features=2 * hidden_channels + num_node_features, out_features=1)

    def forward(self, x, edge_index, edge_weight, pooling_type, batch):
        residual_0 = x
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.bn_conv1(x)
        x = F.dropout(x, p=self.drop_p)
        residual_1 = x
        x = torch.cat([x, residual_0], dim=1)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.bn_conv2(x)
        x = F.dropout(x, p=self.drop_p)
        x = torch.cat([x, residual_1], dim=1)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn_conv3(x)
        # 2. Readout layer
        # Global Pooling (stack different aggregations)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = eval(f'{pooling_type}')(x, batch=batch)
        # Batch norm for embeddings
        x = self.bn1(x)
        self.embeddings = x.detach().to('cpu').numpy()

        # 3. Apply a final classifier
        #x = F.dropout(x, p=self.drop_p)
        x = self.lin(x)
        return x

    def get_embeddings(self):
        return self.embeddings


class GCN(Module):
    # The Class is derived from GCN
    def __init__(self, config, num_node_features):
        hidden_channels = config.training['embedding_size']
        # Default values for embeddings
        self.embeddings = np.ones(2*hidden_channels)
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels, normalize=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.bn1 = BatchNorm1d(num_features=hidden_channels)
        self.lin = Linear(in_features=hidden_channels, out_features=1)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        # Global Pooling (stack different aggregations)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = gmp(x, batch)
        # Batch norm for embeddings
        x = self.bn1(x)
        self.embeddings = x.detach().to('cpu').numpy()

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def get_embeddings(self):
        return self.embeddings
