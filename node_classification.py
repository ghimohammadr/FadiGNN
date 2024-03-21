import argparse
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from torch import tensor 
import time
from APPNP_importance_score import APPNP_PPR
from splits import split
from load_data import load_data
from adjacency import get_adj_sparse
from network import GCN, APP, Transformer
from sklearn.metrics import accuracy_score
import os
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=10)
# Set a seed value (you can use any integer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=int, default=0.85)
    parser.add_argument('--seed_value', type=int, default=42)
    parser.add_argument('--percentages', type=float, default=[10]) # range from 0 to 100
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--name_data', type=str, default='cora') 
    parser.add_argument('--splittion', type=str, default='fullsupervised') 
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=400) 
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed_value)

    data, adj_matrix, args.n_classes, args.num_features = load_data(args.name_data, device)
    adj_matrix_sparse = get_adj_sparse(adj_matrix)

    all_acc = []

    # Get the scores
    scores = APPNP_PPR(args, data.x, data.edge_index, adj_matrix_sparse, device)


    start = time.time()
    for run in range(args.runs):

        print("Run: ", run)

        # Define the model, optimizer, and loss function
        model = GCN(args.num_features, args.n_classes, 64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.reset_parameters()

        # split the data
        data = split(data, args.n_classes, args.splittion)

        best_val_loss = float('inf')
        for percent in args.percentages:


            k = int(percent / 100 * len(scores))
            # select the top ratio percent large values
            _, topk_indices = torch.topk(scores.transpose(0, 1), k, largest=True, sorted=False)
            # Use torch.topk to get the smallest-k values and indices
            _, smallest_k_indices = torch.topk(scores.transpose(0, 1), k, largest=False, sorted=False)

            # get new adjacency matrix
            new_adj = adj_matrix[topk_indices.tolist()[0]][:, topk_indices.tolist()[0]]

            # Filter the original edge_index to only include edges involving important nodes
            new_edge_index = new_adj.to_sparse().coalesce().indices()
            # Create a new feature matrix for important nodes
            new_x = data.x[topk_indices]
            new_x = new_x.squeeze(0)

            new_train_label = data.y[topk_indices.tolist()[0]]
            new_val_label = data.y[topk_indices.tolist()[0]]
            # Define train_mask, val_mask, and test_mask for important nodes
            new_train_mask = data.train_mask[topk_indices.tolist()[0]]
            new_val_mask = data.val_mask[topk_indices.tolist()[0]]

            # Normalization
            if args.normalize_features:
                new_x = F.normalize(new_x, p=2)
                data.x = F.normalize(data.x, p=2)
            
            val_loss_history = []
            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                out, _, _ = model(new_x, new_edge_index)
                loss = F.nll_loss(out[new_train_mask], new_train_label[new_train_mask]) 
                loss.backward()
                optimizer.step()


                model.eval()
                pred, _, _ = model(new_x, new_edge_index)
                val_loss = F.nll_loss(pred[new_val_mask], new_val_label[new_val_mask]).item()

                if val_loss < best_val_loss and epoch > args.epochs // 2:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

                val_loss_history.append(val_loss)
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        break

        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()

        _, logits, _ = model(data.x, data.edge_index)
        # test_acc = int(logits[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
        test_acc = accuracy_score(data.y[data.test_mask], logits.argmax(1).cpu().detach().numpy()[data.test_mask])
        all_acc.append(test_acc)
        print("Accuracy is: ", test_acc)


    end = time.time()
    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
    print('ave_time:', (end-start)/args.runs)

