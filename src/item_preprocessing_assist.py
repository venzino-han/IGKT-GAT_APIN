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


ASSIST_TAGS = 102

def tag_to_vector(tag):
    tag_lsit = [0]*ASSIST_TAGS
    if isinstance(tag, float):
        return np.array(tag_lsit)
    tag_lsit[tag]=1
    return np.array(tag_lsit)




if __name__=="__main__":
    ques_path = "data/assist/questions.csv"
    # question_id,bundle_id,correct_answer,part,tags

    df = pd.read_csv(ques_path)
    df['tag_vector'] = df['tags'].apply(tag_to_vector)
    tag_matrix = df['tag_vector'].tolist()
    tag_matrix = np.stack(tag_matrix, axis=0)
    tag_coo_matrix = np.matmul(tag_matrix, tag_matrix.T)
    print(tag_coo_matrix.sum())

    n_items = 3162

    item_tag_coo_matrix = defaultdict(dict)
    for i in range(n_items):
        for j in range(n_items):
            v = tag_coo_matrix[i][j]
            if v != 0:
                item_tag_coo_matrix[i][j]=v    

    with open('./data/assist/tag_coo_matrix.pkl', 'wb') as pick:
        pickle.dump(item_tag_coo_matrix, pick)

    parts = df['part'].tolist()

    item_part_matrix=defaultdict(dict)
    for i in range(n_items):
        for j in range(n_items):
            if parts[i]==parts[j]:
                item_part_matrix[i][j]=1

    with open('./data/assist/part_matrix.pkl', 'wb') as pick:
        pickle.dump(item_part_matrix, pick)
