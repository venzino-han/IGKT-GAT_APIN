"""IGMC modules"""
import numpy as np
import torch as th
import torch.nn as nn
from dgl.nn.pytorch.conv import EGATConv

class IGAKT(nn.Module):

    def __init__(self, in_nfeats, in_efeats, latent_dim,
                 num_heads=4, edge_dropout=0.2,):
                 
        super(IGAKT, self).__init__()
        self.edge_dropout = edge_dropout
        self.in_nfeats = in_nfeats
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()
        self.convs = th.nn.ModuleList()
        
        self.convs.append(EGATConv(in_node_feats=4,
                                   in_edge_feats=in_efeats,
                                   out_node_feats=latent_dim[0],
                                   out_edge_feats=in_efeats,
                                   num_heads=num_heads))

        for i in range(0, len(latent_dim)-1):
            self.convs.append(EGATConv(in_node_feats=latent_dim[i],
                                    in_edge_feats=in_efeats,
                                    out_node_feats=latent_dim[i+1],
                                    out_edge_feats=in_efeats,
                                    num_heads=num_heads))

        self.lin1 = nn.Linear(2*sum(latent_dim), 128) # concat user, item vector
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, 1)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_parameters(self):
        parameters_dict = {}
        n, p  = self.lin1.named_parameters()
        parameters_dict[n] = p
        n, p  = self.lin2.named_parameters()
        parameters_dict[n] = p
        return parameters_dict


    def forward(self, graph):
        """ graph : subgraph """
        graph = edge_drop(graph, self.edge_dropout, self.training)

        graph.edata['norm'] = graph.edata['edge_mask']
        node_x = graph.ndata['x'].float()

        states = []

        # get user, item idx --> vector
        users = graph.ndata['nlabel'][:, 0] == 1
        items = graph.ndata['nlabel'][:, 1] == 1

        x = node_x # original
        e = graph.edata['efeat']
        
        for conv in self.convs:
            x, _ = conv(graph=graph, nfeats=x, efeats=e)
            x = th.sum(x, dim=1)
            x = self.elu(x)
            states.append(x)

        states = th.cat(states, 1)
        x = th.cat([states[users], states[items]], 1)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)

        return x[:, 0]


def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph