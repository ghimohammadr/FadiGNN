import torch

def get_adj_sparse(adj_matrix):

    
    # add selfloop and normalize
    self_loops = torch.eye(adj_matrix.size(0))  # Create a diagonal matrix with ones
    adj_matrix_with_self_loops = adj_matrix + self_loops

    # Step 1: Calculate the degree matrix
    degree_matrix = adj_matrix_with_self_loops.sum(dim=1)
    # Step 2: Compute the diagonal degree matrix D~^(-1/2)
    diagonal_degree_matrix = torch.diag(1.0 / torch.sqrt(degree_matrix))
    # Step 3: Compute the symmetrically normalized adjacency matrix A~ˆ
    normalized_adjacency = torch.mm(torch.mm(diagonal_degree_matrix, adj_matrix_with_self_loops), diagonal_degree_matrix)

    normalized_adjacency = normalized_adjacency.unsqueeze(0)
    # Convert the adjacency matrix to a PyTorch sparse tensor
    adj_matrix_numpy = normalized_adjacency.numpy()[0]
    adj_matrix_sparse = torch.sparse.FloatTensor(
        torch.LongTensor(adj_matrix_numpy.nonzero()),
        torch.FloatTensor(adj_matrix_numpy[adj_matrix_numpy.nonzero()])
    )


    return adj_matrix_sparse
