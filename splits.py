import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def split(data, num_classes, exp):
    if exp != 'fixed':
        indices = []
        for i in range(num_classes):
            # index = torch.nonzero(torch.eq(data.y, i)).squeeze()
            index = (data.y == i).nonzero().view(-1)
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
