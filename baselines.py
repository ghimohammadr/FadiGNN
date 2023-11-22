import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP, TransformerConv
from torch_geometric.datasets import Planetoid, LastFMAsia
import torch_geometric.transforms as T
import warnings
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp
warnings.filterwarnings("ignore")
from torch import tensor 
import os
import time
from performer_pytorch import FastAttention
from torch_geometric.utils import to_networkx
import networkx as nx
# Set a seed value (you can use any integer)
seed_value = 42
torch.manual_seed(seed_value)
path = "params/"
if not os.path.isdir(path):
    os.mkdir(path)


# Model construction
class fastAPPNPTransformerBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        #assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = 4
        self.embedding_dim = hidden
        self.head_dim = self.embedding_dim // self.num_heads
        
        
        # for convolution
        self.lin1 = nn.Linear(dataset.num_features, self.embedding_dim) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.lin2 = nn.Linear(2*self.embedding_dim, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

        # for MHA
        self.q_proj = nn.Linear(dataset.num_features, self.embedding_dim)
        self.k_proj = nn.Linear(dataset.num_features, self.embedding_dim)
        self.v_proj = nn.Linear(dataset.num_features, self.embedding_dim)
        self.fastattn = FastAttention(dim_heads = self.head_dim, causal = False)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.prop1.reset_parameters()

    def qkv_calculation(self, unsqueezed):
        batch_size, seq_length, embedd_dim = unsqueezed.size()
        q = self.q_proj(unsqueezed)
        k = self.k_proj(unsqueezed)
        v = self.v_proj(unsqueezed)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]

        values = self.fastattn(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)

        return values

    def forward(self, data, edge_index):
        data = F.dropout(data, training=self.training)

        unsqueezed = data.unsqueeze(0)
        squeezed = data
        transembedding = self.qkv_calculation(unsqueezed)
        # embed of conv 
        gnnembedding1 = self.lin1(squeezed)     
        gnnembedding_bein = torch.concat((gnnembedding1,transembedding.squeeze(0)), 1)
        gnnembedding_bein = F.elu(gnnembedding_bein)
        gnnembedding_bein = F.dropout(gnnembedding_bein, training=self.training)
        gnnembedding2 = self.lin2(gnnembedding_bein)
        gnnembedding2 = F.elu(gnnembedding2)
        gnnembedding2 = F.dropout(gnnembedding2, training=self.training)
        gnnembeddingfinal = self.prop1(gnnembedding2, edge_index)

        return F.log_softmax(gnnembeddingfinal, dim=1), gnnembeddingfinal
  
class APP(torch.nn.Module):
    def __init__(self, hidden):
        super(APP, self).__init__()

        self.hidden = hidden

        self.lin1 = nn.Linear(dataset.num_features, self.hidden)
        self.lin2 = nn.Linear(self.hidden, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        
    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1), x

class GCN(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(GCN, self).__init__()

        self.embedding_dim = embedding_dim
        
        self.conv1 = GCNConv(dataset.num_features, self.embedding_dim)
        self.conv2 = GCNConv(self.embedding_dim, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1), x
    
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def splits(data, num_classes, exp):
    if exp != 'fixed':
        indices = []
        for i in range(num_classes):
            index = torch.nonzero(torch.eq(data.y, i)).squeeze()
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        
        if exp == 'fullsupervised':
            train_index = torch.cat([i[:int(0.6*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.6*len(i)):int(0.8*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.8*len(i)):] for i in indices], dim=0)
        elif exp == 'fullfullsupervised':
            train_index = torch.cat([i[:int(0.8*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.8*len(i)):int(0.9*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.9*len(i)):] for i in indices], dim=0)
        elif exp == 'semisupervised':
            train_index = torch.cat([i[:int(0.025*len(i))] for i in indices], dim=0)
            val_index = torch.cat([i[int(0.025*len(i)):int(0.05*len(i))] for i in indices], dim=0)
            test_index = torch.cat([i[int(0.05*len(i)):] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_printoptions(precision=10)

# read the data
name_data = 'LastFMAsia'
# dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset = LastFMAsia(root= '/tmp/' + name_data)
dataset.transform = T.NormalizeFeatures()
data = dataset[0].to(device)
adj_matrix = to_dense_adj(data.edge_index)
# Convert the adjacency matrix to a PyTorch sparse tensor
adj_matrix_numpy = adj_matrix.numpy()[0]
adj_matrix_sparse = torch.sparse.FloatTensor(
    torch.LongTensor(adj_matrix_numpy.nonzero()),
    torch.FloatTensor(adj_matrix_numpy[adj_matrix_numpy.nonzero()])
)

#Full supervised split
n_classes = len(set(np.array(dataset[0].y)))
data = splits(data, n_classes, 'fullsupervised')


# parameters
percentages = [80]
early_stopping = 10
epochs = 300
runs = 3
all_acc = []
all_macro = []



# Convert the Cora dataset to a NetworkX graph
G = to_networkx(data)
# Calculate degree centrality
degree_centralities = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
pagerank_centrality = nx.pagerank(G)

# get the values
value =  degree_centralities.values()
scores = torch.tensor(list(value))  # Convert dict_values to a list


# get adjacency matrix
adj = adj_matrix.squeeze_(0)


start = time.time()
for run in range(runs):

    print("Run: ", run)

    # Define the model, optimizer, and loss function
    model = GCN(64).to(device)
    # model = fastAPPNPTransformerBlock(64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.reset_parameters()

    best_val_loss = float('inf')
    for percent in percentages:

        print("percent: ", percent)

        k = int(percent / 100 * len(scores))
        # select the top ratio percent large values
        topk_values, topk_indices = torch.topk(scores, k, largest=True, sorted=False)

        # Use torch.topk to get the smallest-k values and indices
        smallest_k_values, smallest_k_indices = torch.topk(scores, k, largest=False, sorted=False)


        #topk_indices = smallest_k_indices



        # get new adjacency matrix
        new_adj = adj[topk_indices.tolist()][:, topk_indices.tolist()]

        # Filter the original edge_index to only include edges involving important nodes
        new_edge_index = new_adj.to_sparse().coalesce().indices()
        # Create a new feature matrix for important nodes
        new_x = data.x[topk_indices]
        new_x = new_x.squeeze_(0)

        new_train_label = data.y[topk_indices.tolist()]
        new_val_label = data.y[topk_indices.tolist()]
        # Define train_mask, val_mask, and test_mask for important nodes
        new_train_mask = data.train_mask[topk_indices.tolist()]
        new_val_mask = data.val_mask[topk_indices.tolist()]
        new_test_mask = data.test_mask[topk_indices.tolist()]

        # Normalization
        new_x = F.normalize(new_x, p=2)
        data.x = F.normalize(data.x, p=2)


        
        val_loss_history = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out, _ = model(new_x, new_edge_index)
            loss = F.nll_loss(out[new_train_mask], new_train_label[new_train_mask]) 
            loss.backward()
            optimizer.step()


            model.eval()
            pred, _ = model(new_x, new_edge_index)
            val_loss = F.nll_loss(pred[new_val_mask], new_val_label[new_val_mask]).item()


            # if val_loss < best_val_loss and epoch > epochs // 2:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')


            val_loss_history.append(val_loss)
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
    model.eval()
    _, logits = model(data.x, data.edge_index)
    test_acc = accuracy_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask])
    f1macro = f1_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask], average='macro')
    print("Accuracy and F1 Macro are: ", test_acc, f1macro)
    all_acc.append(test_acc)
    all_macro.append(f1macro)

end = time.time()
print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
print('ave_macro: {:.4f}'.format(np.mean(all_macro)), '+/- {:.4f}'.format(np.std(all_macro)))
print('ave_time:', (end-start)/runs)

