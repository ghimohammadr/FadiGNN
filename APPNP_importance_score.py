# Read the packages
import numpy as np
import torch
import time
import torch
from torch import nn
from network import GCN, APP, Transformer
import warnings
warnings.filterwarnings("ignore")


def Compute_z(first_val, y, adj_mat, alpha):

    az = torch.spmm(adj_mat,y)
    alpha_h = alpha*first_val
    z = (1-alpha)*az + alpha_h

    return y, z

def train_model(epoch, model, optimizer, features, adj_matrix_sparse, first_loss, second_loss, loss_temp, edge_index, args):
    model.train()
    optimizer.zero_grad()
    t = time.time()
    y, _ = model(features, edge_index)

    if epoch == 0:
        first_val = y
        np.save('first_val.npy', first_val.detach().cpu().numpy())
    else:
        first_val = np.load('first_val.npy')

    y, z = Compute_z(torch.tensor(first_val, requires_grad=True), y, adj_matrix_sparse, args.alpha)


    loss1 = nn.MSELoss()(y,z)
    loss2 = y.abs().mean()

    # loss1 = torch.norm(y-z, 2)
    # loss2 = torch.norm(y, 2)

    loss_train = loss1 #- loss2
    first_loss.append(loss1.cpu().detach())
    second_loss.append(loss2.cpu().detach())
    loss_temp.append(loss_train.cpu().detach())
    
    loss_train.backward(retain_graph=True)
    optimizer.step()
    
    return y


def APPNP_PPR(args, features, edge_index, adj_matrix_sparse, device):
    model= GCN(args.num_features, 1, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.reset_parameters()

    
    start_time = time.time()  
    loss_temp=[]
    first_loss = []
    second_loss = []
    
    for ep in range(200):
        y = train_model(ep, model, optimizer, features, adj_matrix_sparse, first_loss, second_loss, loss_temp, edge_index, args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return y