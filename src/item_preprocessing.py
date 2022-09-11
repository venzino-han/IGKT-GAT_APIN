"""
pre-process tag, part 
"""
from collections import defaultdict
import time
import pickle
import numpy as np
import pandas as pd
from copy import copy

from scipy.sparse import coo_matrix, save_npz

import config



def tag_to_vector(tags):
    tag_lsit = [0]*188
    if isinstance(tags, float):
        return np.array(tag_lsit)
    tags = tags.split()
    tags = map(int, tags)
    for tag in tags:
        tag_lsit[tag]=1
    return np.array(tag_lsit)




if __name__=="__main__":
    ques_path = "data/questions.csv"
    # question_id,bundle_id,correct_answer,part,tags

    df = pd.read_csv(ques_path)
    df['tag_vector'] = df['tags'].apply(tag_to_vector)
    tag_matrix = df['tag_vector'].tolist()
    tag_matrix = np.stack(tag_matrix, axis=0)
    tag_coo_matrix = np.matmul(tag_matrix, tag_matrix.T)
    print(tag_coo_matrix.sum())

    n_items = config.TOTAL_EXE
    # us, vs = [], []
    # values = []
    item_tag_coo_matrix = defaultdict(dict)
    for i in range(n_items):
        for j in range(n_items):
            v = tag_coo_matrix[i][j]
            if v != 0:
                item_tag_coo_matrix[i][j]=v
                # us.append(i)
                # vs.append(j)
                # values.append(v)
    

    with open('./data/tag_coo_matrix.pkl', 'wb') as pick:
        pickle.dump(item_tag_coo_matrix, pick)

    # item_tag_coo_matrix = coo_matrix((values, (us, vs)), shape=(n_items, n_items))
    # save_npz('./data/tag_coo_matrix.npz', item_tag_coo_matrix)

    parts = df['part'].tolist()
    # us, vs = [], []
    # values = []
    item_part_matrix=defaultdict(dict)
    for i in range(n_items):
        for j in range(n_items):
            if parts[i]==parts[j]:
                item_part_matrix[i][j]=v
                # us.append(i)
                # vs.append(j)
                # values.append(v)

    with open('./data/part_matrix.pkl', 'wb') as pick:
        pickle.dump(item_part_matrix, pick)

    # item_part_matrix = coo_matrix((values, (us, vs)), shape=(n_items, n_items))
    # save_npz('./data/part_matrix.npz', item_part_matrix)
    # print(item_part_matrix.toarray())
    # print(item_part_matrix.toarray().sum())

    # sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
