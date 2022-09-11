import torch as th
import pandas as pd
import numpy as np
import dgl

from graph import UserExeGraph


#######################
# Subgraph Extraction 
#######################

def map_newid(df, col):
    old_ids = df[col]
    old_id_uniq = old_ids.unique()

    id_dict = {old: new for new, old in enumerate(sorted(old_id_uniq))}
    new_ids = np.array([id_dict[x] for x in old_ids])

    return new_ids


def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.int32)
    x[th.arange(len(idx)), idx] = 1.0
    return x  


#######################
# Subgraph Extraction 
#######################
def get_subgraph_label(graph:dgl.graph,
                       u_node_idx:th.tensor, i_node_idx:th.tensor,
                       u_neighbors:th.tensor, i_neighbors:th.tensor,
                       sample_ratio=1.0,)->dgl.graph:

    nodes = th.cat([u_node_idx, i_node_idx, u_neighbors, i_neighbors], dim=0,) 
    nodes = nodes.type(th.int32)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 
    node_labels = [0,1] + [2]*len(u_neighbors) + [3]*len(i_neighbors)
    subgraph.ndata['nlabel'] = one_hot(node_labels, 4)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph = dgl.add_self_loop(subgraph)
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.float32)
    target_edges = subgraph.edge_ids([0, 1], [1, 0], return_uv=False)
    # target_edges = target_edges.type(th.int64)
    # subgraph.edata['edge_mask'][target_edges] = 0

    timestamps = subgraph.edata['ts']
    standard_ts = timestamps[target_edges.to(th.long)[0]]
    timestamps = th.abs(timestamps - standard_ts.item())
    timestamps = 1 - (timestamps - th.min(timestamps)) / (th.max(timestamps)-th.min(timestamps) + 1e-5)
    subgraph.edata['ts'] = timestamps + 1e-5
    subgraph.remove_edges(target_edges)
    return subgraph    



#######################
# Subgraph Dataset 
#######################

class UserItemDataset(th.utils.data.Dataset):
    def __init__(self, user_item_graph: UserExeGraph, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=100):

        self.g_labels = user_item_graph.labels
        self.graph = user_item_graph.graph
        self.pairs = user_item_graph.user_item_pairs
        self.nid_neghibor_dict = user_item_graph.nid_neghibor_dict

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u_idx, i_idx = self.pairs[idx]
        u_neighbors, i_neighbors = self.nid_neghibor_dict[u_idx.item()], self.nid_neghibor_dict[i_idx.item()]
        u_neighbors, i_neighbors = u_neighbors[-self.max_nodes_per_hop:], i_neighbors[-self.max_nodes_per_hop:]
        u_neighbors = u_neighbors[u_neighbors!=i_idx.item()]
        i_neighbors = i_neighbors[i_neighbors!=u_idx.item()]
        subgraph = get_subgraph_label(graph = self.graph,
                                      u_node_idx=u_idx.unsqueeze(0), 
                                      i_node_idx=i_idx.unsqueeze(0), 
                                      u_neighbors=u_neighbors, 
                                      i_neighbors=i_neighbors,           
                                      sample_ratio=self.sample_ratio,
                                    )

        if 'feature' in subgraph.edata.keys():
            masked_feat = th.mul(subgraph.edata['feature'], th.unsqueeze(subgraph.edata['edge_mask'],1))
            subgraph.edata['feature']= masked_feat
        
        g_label = self.g_labels[idx]
        return subgraph, g_label


def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label

NUM_WORKERS = 8

def get_dataloader(data_path, batch_size=32, feature_path=None):

    train_df = pd.read_csv(f'{data_path}_train.csv')
    valid_df = pd.read_csv(f'{data_path}_valid.csv')
    test_df = pd.read_csv(f'{data_path}_test.csv')

    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([valid_df, test_df])

    train_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                edge_feature_from=feature_path,
                                df=train_df,
                                edge_idx_range=(0, len(train_df)))

    train_dataset = UserItemDataset(user_item_graph=train_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)


    valid_graph = UserItemGraph(label_col='rating',
                            user_col='user_id',
                            item_col='item_id',
                            edge_feature_from=feature_path,
                            df=valid_df,
                            edge_idx_range=(len(train_df), len(valid_df)))

    valid_dataset = UserItemDataset(user_item_graph=valid_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    valid_loader = th.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)


    test_graph = UserItemGraph(label_col='rating',
                            user_col='user_id',
                            item_col='item_id',
                            edge_feature_from=feature_path,
                            df=test_df,
                            edge_idx_range=(len(valid_df), len(test_df)))

    test_dataset = UserItemDataset(user_item_graph=test_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)

    return train_loader, valid_loader, test_loader