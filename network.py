import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP
import warnings
warnings.filterwarnings("ignore")
from performer_pytorch import FastAttention

class APP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden):
        super(APP, self).__init__()

        self.hidden = hidden

        self.lin1 = nn.Linear(num_features, self.hidden)
        self.lin2 = nn.Linear(self.hidden, num_classes)
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

        return F.log_softmax(x, dim=1), x, F.softmax(x, dim=0)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x, F.softmax(x, dim=0)

class Transformer(nn.Module):
    def __init__(self, num_features, num_classes, hidden):
        super().__init__()
        #assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = 4
        self.embedding_dim = hidden
        self.head_dim = self.embedding_dim // self.num_heads

        # for MHA
        self.q_proj = nn.Linear(num_features, self.embedding_dim)
        self.k_proj = nn.Linear(num_features, self.embedding_dim)
        self.v_proj = nn.Linear(num_features, self.embedding_dim)
        self.fastattn = FastAttention(dim_heads = self.head_dim, causal = False)
        self.lin1 = nn.Linear(self.embedding_dim, num_classes)

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.lin1.reset_parameters()

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

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)

        x = self.qkv_calculation(x.unsqueeze(0))
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x.squeeze(0))

        return F.log_softmax(x, dim=1), x, F.softmax(x, dim=0)
    
