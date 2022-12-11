import pickle
import numpy as np
import pandas as pd

import torch as th
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix

from data_generator import get_subgraph_label

import dgl

import config

def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.int32)
    x[th.arange(len(idx)), idx] = 1.0
    return x  


class KT_Sequence_Graph(Dataset):
    def __init__(self, user_groups, item_groups, df, part_matrix, tag_coo_matrix, seq_len):
        self.user_seq_dict = {}
        self.seq_len = seq_len
        self.user_ids = []
        self.part_matrix, self.tag_coo_matrix = part_matrix, tag_coo_matrix,

        self.user_id_set = set()

        uids = []
        eids = []
        correctness = []

        # get user seqs
        for user_id in user_groups.index:
            self.user_id_set.add(user_id)
            # "user_id", "content_id", "answered_correctly", "timestamp", 'part'
            c_id, ans_c, ts, part = user_groups[user_id]

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
                        c_id[:initial], part[:initial], ans_c[:initial], ts[:initial]
                    )
                chunks = len(c_id)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.user_seq_dict[f"{user_id}_{c+1}"] = (
                        c_id[start:end], part[start:end], ans_c[start:end], ts[start:end]
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.user_seq_dict[f"{user_id}"] = (c_id, part, ans_c, ts)
        
        self.item_seq_dict = {}
        for user_seq_id in self.user_ids:
            user_seq = self.user_seq_dict[user_seq_id]
            target_cid = user_seq[0]
            target_cid = target_cid[-1]
            u_id, ans_c, ts, part = item_groups[target_cid]
            n = self.seq_len #*2
            if n > len(u_id):
                n =len(u_id)
            indices = np.random.choice(len(u_id), n, replace=False)
            self.item_seq_dict[user_seq_id] = u_id[indices]

        # build user-exe matrix
        uids = df['user_id']
        eids = df['content_id']
        correctness = df['answered_correctly']
        ts = df['timestamp']
        ts=(ts-ts.min())/(ts.max()-ts.min())
        num_user = max(uids)+1

        print(num_user)
        print(num_user+config.ASSIST_EXE)

        uids += config.ASSIST_EXE
        num_nodes = num_user+config.ASSIST_EXE

        src_nodes = np.concatenate((uids, eids))
        dst_nodes = np.concatenate((eids, uids))
        correctness = np.concatenate((correctness, correctness))
        ts = np.concatenate((ts, ts))
        print(len(src_nodes), len(dst_nodes), len(correctness))
        user_exe_matrix = coo_matrix((correctness, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        
        # build graph 
        self.graph = dgl.from_scipy(sp_mat=user_exe_matrix, idtype=th.int32)
        self.graph.ndata['node_id'] = th.tensor(list(range(num_nodes)), dtype=th.int32)
        self.graph.edata['label'] = th.tensor(correctness, dtype=th.float32)
        self.graph.edata['etype'] = th.tensor(correctness, dtype=th.int32)
        self.graph.edata['ts'] = th.tensor(ts, dtype=th.float32)
        ts_max = self.graph.edata['ts'].max()


        src, dst, etypes = [], [], []

        print('--------------------------- node degree before')
        print(self.graph.in_degrees().float().mean())
        print('--------------------------- node degree after')

        print('start part')
        for i in self.part_matrix.keys():
            for j in self.part_matrix[i].keys():
                if i==j: continue
                part = self.part_matrix[i][j]
                if part > 0:
                    src.append(i)
                    dst.append(j)
                    etypes.append(2)

        print('start tag')
        for i in self.tag_coo_matrix.keys():
            for j in self.tag_coo_matrix[i].keys():
                if i==j: continue
                tag_coo = self.tag_coo_matrix[i][j]
                if tag_coo == 1:
                    src.append(i)
                    dst.append(j)
                    etypes.append(2)
                elif tag_coo == 2:
                    src.append(i)
                    dst.append(j)
                    etypes.append(3)
                elif tag_coo >= 3:
                    src.append(i)
                    dst.append(j)
                    etypes.append(4)
               
        print('start adding edges')
        n_edges =  len(etypes)

        # LIMIT = 100000

        # etypes=etypes[:LIMIT]
        edata = {
            'etype': th.tensor(np.array(etypes), dtype=th.int32),
            'label': th.tensor(np.array([1.]*n_edges), dtype=th.float32),
            'ts': th.tensor(np.array([ts_max]*n_edges), dtype=th.float32),
        }

        self.graph.add_edges(src, dst, data=edata)
        print(self.graph.in_degrees().float().mean())

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_seq_id = self.user_ids[index]
        c_id, part, ans_c, ts = self.user_seq_dict[user_seq_id]
        u_id = self.item_seq_dict[user_seq_id]
        seq_len = len(c_id)

        #build graph
        label = ans_c[1:] - 1
        label = np.clip(label, 0, 1)
        
        target_item_id = c_id[-1]
        label = label[-1]

        #build graph
        u_idx, i_idx = int(user_seq_id.split('_')[0])+config.ASSIST_EXE, target_item_id
        u_neighbors, i_neighbors = u_id+config.ASSIST_EXE, c_id[:-1]
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

def get_dataloader_assist(batch_size, num_workers=8, seq_len=64):
    with open('./data/assist/part_matrix.pkl', 'rb') as pick:
        part_matrix = pickle.load(pick)
    with open("data/assist/tag_coo_matrix.pkl", 'rb') as pick:
        tag_coo_matrix = pickle.load(pick)


    train_df = pd.read_csv('data/assist/train_df.csv')
    test_df = pd.read_csv('data/assist/test_df.csv')

    with open("data/assist/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open("data/assist/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)
    
    train_seq_graph = KT_Sequence_Graph(train_user_group, train_item_group, df=train_df, 
                                           part_matrix=part_matrix, tag_coo_matrix=tag_coo_matrix, seq_len=seq_len)
    train_loader = DataLoader(train_seq_graph, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    with open("data/assist/val_user_group.pkl.zip", 'rb') as pick:
        val_user_group = pickle.load(pick)
    with open("data/assist/val_item_group.pkl.zip", 'rb') as pick:
        val_item_group = pickle.load(pick)

    test_seq_graph = KT_Sequence_Graph(val_user_group, val_item_group, df=test_df, 
                                          part_matrix=part_matrix, tag_coo_matrix=tag_coo_matrix, seq_len=seq_len)
    test_loader = DataLoader(test_seq_graph, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_data, pin_memory=True)

    return train_loader, test_loader



if __name__=="__main__":
    with open("data/assist/train_user_group.pkl.zip", 'rb') as pick:
        train_user_group = pickle.load(pick)
    with open("data/assist/train_item_group.pkl.zip", 'rb') as pick:
        train_item_group = pickle.load(pick)

    train_loader, test_loader = get_dataloader(batch_size=1)

    for batch, label in train_loader:
        print(batch)
        print(label.shape)
        print(label)
        break
