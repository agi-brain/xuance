import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_geometric.data import Data, DataLoader

batch_size = 3
num_nodes = 4
num_feat = 5

x = torch.rand((batch_size, num_nodes, num_feat))
adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes))
conv = GCNConv(num_feat, 6)

datalist = [
    Data(
        x=x[i],
        edge_index=dense_to_sparse(adj[i])[0],
        edge_attr=dense_to_sparse(adj[i])[1],
    )
    for i in range(batch_size)
]
loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)

batch = next(iter(loader))
out = conv(batch.x, batch.edge_index)
out, mask = to_dense_batch(out, batch.batch)
