from torch_geometric.datasets import Planetoid, LastFMAsia, CitationFull, AttributedGraphDataset, WebKB, Actor, WikipediaNetwork
import torch_geometric.transforms as T
import torch
from torch import nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_adj, coalesce, remove_self_loops, to_undirected
import numpy as np
# read the data
def load_data(name_data, device):
    if name_data == 'dblp':
        dataset = CitationFull(root='./dataset', name=name_data)
    elif name_data == 'Cora_ML':
        dataset = CitationFull(root='./dataset', name=name_data)
    elif name_data == 'BlogCatalog':
        dataset = AttributedGraphDataset(root='./dataset', name=name_data)
    elif name_data in ['texas', 'wisconsin']:
        dataset = WebKB(root='./dataset', name=name_data)
    elif name_data == 'lastfmasia':
        dataset = LastFMAsia(root='./dataset')
    else:
        dataset = Planetoid(root='./dataset', name=name_data)
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0].to(device)
    adj_matrix = torch.squeeze(to_dense_adj(data.edge_index), dim=0)


    if name_data in ['texas', 'wisconsin']:
        # make undirect
        x ,y = data.x, data.y
        scaler = StandardScaler()
        x = torch.from_numpy(scaler.fit_transform(x.numpy()))
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index = coalesce(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data = Data(x=x, edge_index=data.edge_index, y=y)

    n_classes = len(set(np.array(dataset[0].y)))


    return data, adj_matrix, n_classes, dataset.num_features
