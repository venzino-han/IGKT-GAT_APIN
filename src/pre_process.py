"""
pre-processing ednet
"""


import time
import pickle
import numpy as np
import pandas as pd
from utils import get_time_lag

from copy import copy

data_type ={
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8"
}

def group_seq(df, groupby_key, cols, save_path):
    # build group : user_id(item_id) - seq
    # save as file
    cols = copy(cols)
    cols.remove(groupby_key)
    print(cols)
    group = df.groupby(groupby_key).apply(lambda df: tuple([df[c].values for c in cols]))
    with open(save_path, 'wb') as pick:
        pickle.dump(group, pick)
    del group, df
    return

def pre_process(train_path, ques_path, row_start=30e6, num_rows=30e6, split_ratio=0.8):
    print("Start pre-process")
    t_s = time.time()

    Features = ["timestamp", "user_id", "content_id", "content_type_id", "task_container_id", "user_answer", 
                "answered_correctly", "prior_question_elapsed_time", "prior_question_had_explanation"]
    df = pd.read_csv(train_path)[Features]
    df.index = df.index.astype('uint32')

    # shift prior elapsed_time and had_explanation to make current elapsed_time and had_explanation
    df = df[df.content_type_id == 0].reset_index()
    df["prior_question_elapsed_time"].fillna(0, inplace=True)
    df["prior_question_elapsed_time"] /= 1000 # convert to sec
    df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].clip(0, 300)
    df["prior_question_had_explanation"].fillna(False, inplace=True)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].astype('int8')
    
    # get time_lag feature
    print("Start compute time_lag")
    time_dict = get_time_lag(df)
    # with open("time_dict.pkl.zip", 'wb') as pick:
    #     pickle.dump(time_dict, pick)
    print("Complete compute time_lag")
    print("====================")

    df = df.sort_values(by=["timestamp"])
    # train_df.drop("timestamp", axis=1, inplace=True)
    # train_df.drop("viretual_time_stamp", axis=1, inplace=True)

    print("Start merge dataframe")
    # merge with question dataframe to get part feature
    ques_df = pd.read_csv(ques_path)[["question_id", "part"]]
    df = df.merge(ques_df, how='left', left_on='content_id', right_on='question_id')
    df.drop(["question_id"], axis=1, inplace=True)
    df["part"] = df["part"].astype('uint8')
    print(df.head(10))
    print("Complete merge dataframe")
    print("====================")

    # plus 1 for cat feature which starts from 0
    df["content_id"] += 1
    df["task_container_id"] += 1
    df["answered_correctly"] += 1
    df["prior_question_had_explanation"] += 1
    df["user_answer"] += 1

    Train_features = ["user_id", "content_id", "part", "task_container_id", "time_lag", "prior_question_elapsed_time",
                      "answered_correctly", "prior_question_had_explanation", "user_answer", "timestamp"]

    if num_rows == -1:
        num_rows = df.shape[0]
    row_start = 0
    num_rows = df.shape[0]
    # df = df.iloc[int(row_start):int(row_start+num_rows)]
    val_df = df[int(num_rows*split_ratio):]
    train_df = df[:int(num_rows*split_ratio)]

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    # Check data balance
    num_new_user = val_df[~val_df["user_id"].isin(train_df["user_id"])]["user_id"].nunique()
    num_new_content = val_df[~val_df["content_id"].isin(train_df["content_id"])]["content_id"].nunique()
    train_content_id = train_df["content_id"].nunique()
    train_part = train_df["part"].nunique()
    train_correct = train_df["answered_correctly"].mean()-1
    val_correct = val_df["answered_correctly"].mean()-1
    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}/ Number of part {}".format(train_content_id, train_part))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")

    print("Start train and Val grouping")

    df.to_csv('data/test_df.csv')
    train_df.to_csv('data/train_df.csv')

    group_seq(df=train_df, groupby_key="user_id", cols=Train_features, save_path="data/train_user_group.pkl.zip")
    group_seq(df=train_df, groupby_key="content_id", cols=Train_features, save_path="data/train_item_group.pkl.zip")

    group_seq(df=val_df, groupby_key="user_id", cols=Train_features, save_path="data/val_user_group.pkl.zip")
    group_seq(df=df, groupby_key="content_id", cols=Train_features, save_path="data/val_item_group.pkl.zip")
    
    print("Complete pre-process, execution time {:.2f} s".format(time.time()-t_s))

if __name__=="__main__":
    train_path = "data/train_30m.csv"
    ques_path = "data/questions.csv"
    # be aware that appropriate range of data is required to ensure all questions 
    # are in the training set, or LB score will be much lower than CV score
    # Recommend to user all of the data.
    pre_process(train_path, ques_path, 0, -1, 0.8)