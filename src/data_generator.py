# from distutils.command.config import config
# import gc
import pickle
import random
import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import coo_matrix

import dgl

import config

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
                       )->dgl.graph:
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

    # timestamps = subgraph.edata['ts']
    # standard_ts = timestamps[target_edges.to(th.long)[0]]
    # timestamps = th.abs(timestamps - standard_ts.item())
    # timestamps = 1 - (timestamps - th.min(timestamps)) / (th.max(timestamps)-th.min(timestamps) + 1e-5)
    # subgraph.edata['ts'] = timestamps + 1e-5
    subgraph.remove_edges(target_edges)
    return subgraph


class Riiid_Sequence_Graph(Dataset):
    def __init__(self, user_groups, item_groups, df, seq_len):
        self.user_seq_dict = {}
        self.seq_len = seq_len
        self.user_ids = []

        self.user_id_set = set()

        uids = []
        eids = []
        correctness = []

        # get user seqs
        for user_id in user_groups.index:
            self.user_id_set.add(user_id)
            c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = user_groups[user_id]

            n = len(c_id)
            uids.extend([user_id]*n)
            eids.extend(c_id)
            correctness.extend(ans_c)

            if len(c_id) < 2:
                continue

            if len(c_id) > self.seq_len:
                initial = len(c_id) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.user_seq_dict[f"{user_id}_0"] = (
                        c_id[:initial], part[:initial], t_c_id[:initial], t_lag[:initial], 
                        q_et[:initial], ans_c[:initial], q_he[:initial], u_ans[:initial],
                        ts[:initial]
                    )
                chunks = len(c_id)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.user_seq_dict[f"{user_id}_{c+1}"] = (
                        c_id[start:end], part[start:end], t_c_id[start:end], t_lag[start:end], 
                        q_et[start:end], ans_c[start:end], q_he[start:end], u_ans[start:end],
                        ts[start:end]
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.user_seq_dict[f"{user_id}"] = (c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts)
        
        self.item_seq_dict = {}
        for user_seq_id in self.user_ids:
            user_seq = self.user_seq_dict[user_seq_id]
            target_cid = user_seq[0]
            target_cid = target_cid[-1]
            u_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = item_groups[target_cid]
            n = self.seq_len
            if n > len(u_id):
                n =len(u_id)
            indices = np.random.choice(len(u_id), n, replace=False)
            self.item_seq_dict[user_seq_id] = u_id[indices]

        # build user-exe matrix
        # TODO: build matrix with train + val
        uids = df['user_id']
        eids = df['content_id']
        correctness = df['answered_correctly']
        num_user = max(uids)+1
        print(num_user)
        print(num_user+config.TOTAL_EXE)
        # uids, eids = np.array(uids), np.array(eids)
        # correctness = np.array(correctness)

        uids += config.TOTAL_EXE
        num_nodes = num_user+config.TOTAL_EXE

        src_nodes = np.concatenate((uids, eids))
        dst_nodes = np.concatenate((eids, uids))
        correctness = np.concatenate((correctness, correctness))
        print(len(src_nodes), len(dst_nodes), len(correctness))
        user_exe_matrix = coo_matrix((correctness, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        
        # build graph 
        self.graph = dgl.from_scipy(sp_mat=user_exe_matrix, idtype=th.int32)
        self.graph.ndata['node_id'] = th.tensor(list(range(num_nodes)), dtype=th.int32)
        self.graph.edata['label'] = th.tensor(correctness, dtype=th.float32)
        self.graph.edata['etype'] = th.tensor(correctness, dtype=th.int32)

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_seq_id = self.user_ids[index]
        c_id, p, t_c_id, t_lag, q_et, ans_c, q_he, u_ans, ts = self.user_seq_dict[user_seq_id]
        u_id = self.item_seq_dict[user_seq_id]
        seq_len = len(c_id)

        #build graph
        # content_ids = content_ids[1:]
        # parts = parts[1:]
        # task_container_ids = task_container_ids[1:]
        # time_lag = time_lag[1:]
        # ques_elapsed_time = ques_elapsed_time[1:]
        label = ans_c[1:] - 1
        label = np.clip(label, 0, 1)
        
        answer_correct = ans_c[:-1]
        # ques_had_explian = ques_had_explian[1:]
        user_answer = u_ans[:-1]

        target_item_id = c_id[-1]
        label = label[-1]

        #build graph
        u_idx, i_idx = int(user_seq_id.split('_')[0])+config.TOTAL_EXE, target_item_id
        u_neighbors, i_neighbors = u_id+config.TOTAL_EXE, c_id[:-1]
        u_neighbors = u_neighbors[u_neighbors!=i_idx]
        i_neighbors = i_neighbors[i_neighbors!=u_idx]

        subgraph = get_subgraph_label(graph = self.graph,
                                      u_node_idx=th.tensor([u_idx]), 
                                      i_node_idx=th.tensor([i_idx]), 
                                      u_neighbors=th.tensor(u_neighbors), 
                                      i_neighbors=th.tensor(i_neighbors),           
                                    )

        return subgraph, th.tensor(label, dtype=th.float32)


def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label

def get_dataloader(batch_size, num_workers=8, seq_len=64):
    train_df = pd.read_csv('data/train_df.csv')
    test_df = pd.read_csv('data/test_df.csv')

    with open("data/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open("data/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = Riiid_Sequence_Graph(train_user_group, train_item_group, df=train_df, seq_len=seq_len)
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    with open("data/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open("data/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = Riiid_Sequence_Graph(val_user_group, val_item_group, df=test_df, seq_len=seq_len)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader



if __name__=="__main__":
    with open("data/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open("data/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    # with open("val_user_group.pkl.zip", 'rb') as pick:
    #     val_user_group = pickle.load(pick)
    # with open("val_item_group.pkl.zip", 'rb') as pick:
    #     val_item_group = pickle.load(pick)
    seq_graph = Riiid_Sequence_Graph(train_user_group, train_item_group, seq_len=64)
    print(seq_graph.graph)
    for subg, label in seq_graph:
        print(subg)
        print(label)
        break

