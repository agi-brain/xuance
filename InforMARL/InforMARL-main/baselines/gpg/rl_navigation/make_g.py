import pdb
import dgl
import torch


def build_graph(adj):
    g = dgl.DGLGraph()
    num_agents = adj.shape[0]

    g.add_nodes(num_agents)
    edge_list = []
    for i in range(0, num_agents):
        for j in range(0, num_agents):
            if adj[i][j] > 0:
                edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    g.set_e_initializer(dgl.init.zero_initializer)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata["feat"] = torch.eye(num_agents)

    return g


def build_graph_dynamic(adj):
    g = dgl.DGLGraph()
    num_agents = adj.shape[0]

    g.add_nodes(num_agents)
    edge_list = []
    for i in range(0, num_agents):
        for j in range(0, num_agents):
            if adj[i][j] == 1:
                edge_list.append((i, j))
    try:
        src, dst = tuple(zip(*edge_list))
        pass
    except Exception as e:
        pdb.set_trace()
        raise

    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    g.set_e_initializer(dgl.init.zero_initializer)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata["feat"] = torch.eye(num_agents)

    return g
