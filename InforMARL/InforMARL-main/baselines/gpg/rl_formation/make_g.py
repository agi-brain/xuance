import pdb
import dgl
import torch


def build_graph(env):
    g = dgl.DGLGraph()
    num_ag = env.n_agents

    g.add_nodes(num_ag)
    adj_matrix = env.get_connectivity(env.x)
    edge_list = []
    for i in range(0, num_ag):
        for j in range(0, num_ag):
            if adj_matrix[i][j] > 0:
                edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    g.set_e_initializer(dgl.init.zero_initializer)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata["feat"] = torch.eye(num_ag)
    # g.set_n_initializer(dgl.init.zero_initializer)

    return g


def build_graph_dynamic(adj_matrix, num_ag):
    g = dgl.DGLGraph()
    g.add_nodes(num_ag)

    edge_list = []

    for i in range(0, num_ag):
        for j in range(0, num_ag):
            if adj_matrix[i][j] == 1:
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

    g.ndata["feat"] = torch.eye(num_ag)
    g.set_e_initializer(dgl.init.zero_initializer)
    g.set_n_initializer(dgl.init.zero_initializer)

    return g
