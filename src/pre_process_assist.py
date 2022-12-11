"""
pre-processing assist 
"""

import time
import pandas as pd
from pre_process import group_seq


data_type ={
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "answered_correctly": "int8",
}

def pre_process(train_path, row_start=30e6, num_rows=30e6, split_ratio=0.8):
    print("Start pre-process")
    t_s = time.time()

    Train_features = ["user_id", "content_id", "answered_correctly", "timestamp", 'part']
    df = pd.read_csv(train_path)[Train_features]
    df.index = df.index.astype('uint32')


    # get time_lag feature
    # print("Start compute time_lag")
    # time_dict = get_time_lag(df)
    # print("Complete compute time_lag")
    # print("====================")

    df = df.sort_values(by=["timestamp"])

    print("Start merge dataframe")
    # merge with question dataframe to get part feature
    df["part"] = df["part"].astype('uint8')
    print(df.head(10))
    print("Complete merge dataframe")
    print("====================")

    # plus 1 for cat feature which starts from 0
    df["content_id"] += 1
    df["answered_correctly"] += 1

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

    df.to_csv('data/assist/test_df.csv')
    train_df.to_csv('data/assist/train_df.csv')

    group_seq(df=train_df, groupby_key="user_id", cols=Train_features, save_path="data/assist/train_user_group.pkl.zip")
    group_seq(df=train_df, groupby_key="content_id", cols=Train_features, save_path="data/assist/train_item_group.pkl.zip")

    group_seq(df=val_df, groupby_key="user_id", cols=Train_features, save_path="data/assist/val_user_group.pkl.zip")
    group_seq(df=df, groupby_key="content_id", cols=Train_features, save_path="data/assist/val_item_group.pkl.zip")
    
    print("Complete pre-process, execution time {:.2f} s".format(time.time()-t_s))

if __name__=="__main__":
    train_path = "data/assist/assist.csv"
    pre_process(train_path, 0, -1, 0.8)