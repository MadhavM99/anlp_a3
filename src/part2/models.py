import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
from torch_geometric.utils import dropout_edge, to_undirected


def root_index(ptr):
    return ptr[:-1]


class RootReadout(nn.Module):
    def __init__(self, in_dim, graph_dim, hidden, num_classes, dropout):
        super().__init__()
        self.root_proj = nn.Linear(in_dim, hidden)
        self.fuse = nn.Linear(graph_dim + hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, h, batch, ptr):
        graph = global_mean_pool(h, batch)
        root = F.relu(self.root_proj(x[root_index(ptr)]))
        z = F.relu(self.fuse(torch.cat([graph, root], dim=-1)))
        z = F.dropout(z, self.dropout, self.training)
        return self.out(z)


class RootMLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, ptr):
        h = F.relu(self.lin1(x[root_index(ptr)]))
        h = F.relu(self.lin2(h))
        h = F.dropout(h, self.dropout, self.training)
        return self.out(h)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = RootReadout(in_dim, hidden, hidden, num_classes, dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, ptr):
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        h = F.dropout(x, self.dropout, self.training)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.head(x, h, batch, ptr)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean", root_weight=True)
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean", root_weight=True)
        self.head = RootReadout(in_dim, hidden, hidden, num_classes, dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, ptr):
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        h = F.dropout(x, self.dropout, self.training)
        h = F.normalize(F.relu(self.conv1(h, edge_index)), p=2, dim=-1)
        h = F.dropout(h, self.dropout, self.training)
        h = F.normalize(F.relu(self.conv2(h, edge_index)), p=2, dim=-1)
        return self.head(x, h, batch, ptr)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, heads, gat_dropout):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=gat_dropout)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False, dropout=gat_dropout)
        self.head = RootReadout(in_dim, hidden, hidden, num_classes, dropout)
        self.gat_dropout = gat_dropout

    def forward(self, x, edge_index, batch, ptr):
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        h = F.dropout(x, self.gat_dropout, self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, self.gat_dropout, self.training)
        h = F.elu(self.conv2(h, edge_index))
        return self.head(x, h, batch, ptr)


class BiGCNBranch(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden + in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, ptr):
        root = root_index(ptr)[batch]
        h1 = self.conv1(x, edge_index)
        h = F.relu(torch.cat([h1, x[root]], dim=-1))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = torch.cat([h, h1[root]], dim=-1)
        return global_mean_pool(h, batch)


class BiGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, edge_drop):
        super().__init__()
        self.top_down = BiGCNBranch(in_dim, hidden, hidden, dropout)
        self.bottom_up = BiGCNBranch(in_dim, hidden, hidden, dropout)
        self.out = nn.Linear(4 * hidden, num_classes)
        self.edge_drop = edge_drop

    def forward(self, x, edge_index, batch, ptr):
        td_edges, _ = dropout_edge(edge_index, p=self.edge_drop, training=self.training)
        bu_edges, _ = dropout_edge(edge_index.flip(0), p=self.edge_drop, training=self.training)
        td = self.top_down(x, td_edges, batch, ptr)
        bu = self.bottom_up(x, bu_edges, batch, ptr)
        return self.out(torch.cat([td, bu], dim=-1))


def build_model(name, in_dim, cfg, num_classes=2):
    hidden, dropout = cfg["hidden"], cfg["dropout"]
    if name == "mlp":
        return RootMLP(in_dim, hidden, num_classes, dropout)
    if name == "gcn":
        return GCN(in_dim, hidden, num_classes, dropout)
    if name == "sage":
        return GraphSAGE(in_dim, hidden, num_classes, dropout)
    if name == "gat":
        return GAT(in_dim, hidden, num_classes, dropout, cfg["gat_heads"], cfg["gat_dropout"])
    if name == "bigcn":
        return BiGCN(in_dim, hidden, num_classes, dropout, cfg["bigcn_edge_drop"])
    raise ValueError(f"unknown model {name}")
