import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_time_lag(df):
    """
    Compute time_lag feature, same task_container_id shared same timestamp for each user
    """
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["user_id", "timestamp", "task_container_id"]].values):
        user_id = row[0]
        timestamp = row[1]
        container_id = row[2]
        if user_id not in time_dict:
            time_lag[idx] = 0
            time_dict[user_id] = [timestamp, container_id, 0] # last_timestamp, last_task_container_id, last_lagtime
        else:
            if container_id == time_dict[user_id][1]:
                time_lag[idx] = time_dict[user_id][2]
            else:
                time_lag[idx] = timestamp - time_dict[user_id][0]
                time_dict[user_id][0] = timestamp
                time_dict[user_id][1] = container_id
                time_dict[user_id][2] = time_lag[idx]

    df["time_lag"] = time_lag/1000/60 # convert to miniute
    df["time_lag"] = df["time_lag"].clip(0, 1440) # clip to 1440 miniute which is one day
    return time_dict



def convert_newid(origin_id:int, id_dict:dict, max_id:int):

    if origin_id in id_dict :
        new_id = id_dict.get(origin_id)
    else:
        id_dict[origin_id] = max_id
        new_id = max_id
        max_id += 1

    return new_id, id_dict, max_id


def reset_id(df, reset_col,  cols):

    user_id_dict, item_id_dict, user_ids, item_ids, user_id_max, item_id_max  = {}, {}, [], [], 0, 0        
    
    for i in tqdm(range(len(df))):
        origin_user_id = df[reset_col].iloc[i]
        # origin_item_id = df[item_col].iloc[i]
        new_user_id, user_id_dict, user_id_max = convert_newid(origin_user_id, user_id_dict, user_id_max)
        # new_item_id, item_id_dict, item_id_max = convert_newid(origin_item_id, item_id_dict, item_id_max)

        user_ids.append(new_user_id)
        # item_ids.append(new_item_id)

    df[reset_col] = user_ids
    # df[item_col] = item_ids

    df = df[cols]
    
    return df

import logging

def get_logger(name, path):

    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)
    
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger



from easydict import EasyDict
import yaml


def get_args_from_yaml(yaml_path):

    with open('train_configs/common_configs.yaml') as f:
        common_cfgs = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = common_cfgs['dataset']
    model_cfg = common_cfgs['model']
    train_cfg = common_cfgs['train']

    with open(yaml_path) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    exp_data_cfg = cfgs.get('dataset', dict())
    exp_model_cfg = cfgs.get('model', dict())
    exp_train_cfg = cfgs.get('train', dict())

    for k, v in exp_data_cfg.items():
        data_cfg[k] = v
    for k, v in exp_model_cfg.items():
        model_cfg[k] = v
    for k, v in exp_train_cfg.items():
        train_cfg[k] = v

    args = EasyDict(
        {   
            'key': cfgs['key'],

            'dataset': data_cfg['name'],
            'keywords': data_cfg.get('keywords'),
            'keyword_edge_k': data_cfg.get('keyword_edge_k'),
            'additional_feature': data_cfg.get('additional_feature'),
            'max_seq': data_cfg.get('max_seq'),
            
            # model configs
            'model_type': model_cfg['type'],
            'hop': model_cfg['hop'],
            'in_nfeats': model_cfg.get('in_nfeats'),
            'out_nfeats': model_cfg.get('out_nfeats'),
            'in_efeats': model_cfg.get('in_efeats'),
            'out_efeats': model_cfg.get('out_efeats'),
            'num_heads': model_cfg.get('num_heads'),
            'node_features': model_cfg.get('node_features'),
            'parameters': model_cfg.get('parameters'),
            'num_relations': model_cfg.get('num_relations', 5),
            'edge_dropout': model_cfg['edge_dropout'],

            'latent_dims': model_cfg.get('latent_dims'), # baseline model

            #train configs
            'device':train_cfg['device'],
            'log_dir': train_cfg['log_dir'],
            'log_interval': train_cfg.get('log_interval'),
            'train_lrs': [ float(lr) for lr in train_cfg.get('learning_rates') ],
            'train_epochs': train_cfg.get('epochs'),
            'batch_size': train_cfg['batch_size'],
            'weight_decay': train_cfg.get('weight_decay', 0),
            'lr_decay_step': train_cfg.get('lr_decay_step'),
            'lr_decay_factor': train_cfg.get('lr_decay_factor'),

        }
    )

    return args

import numpy as np
import torch as th

def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)
    
def feature_evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device), batch[1].to(device))
        labels = batch[2].to(device)
        mse += ((preds - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)