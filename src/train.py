import math, copy

import dgl
import pandas as pd
import numpy as np
import pickle as pkl

import torch as th
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import get_logger, get_args_from_yaml
from data_generator import get_dataloader
import config


from models.igmc import IGMC

def adj_rating_reg(model):
    arr_loss = 0
    for conv in model.convs:
        weight = conv.parameters().view(conv.num_bases, conv.in_feat * conv.out_feat)
        weight = th.matmul(conv.w_comp, weight).view(conv.num_rels, conv.in_feat, conv.out_feat)
        arr_loss += th.sum((weight[1:, :, :] - weight[:-1, :, :])**2)
    return arr_loss

def evaluate(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())
    
    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, val_acc


def train_epoch(model, loss_fn, optimizer, loader, device, logger, log_interval):
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(inputs)
        loss = loss_fn(preds, labels).mean() #+ 0.001 * adj_rating_reg(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)


NUM_WORKER = 16
def train(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    train_loader, test_loader = get_dataloader(
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER,
                                 seq_len=args.max_seq
                                 )

    ### prepare data and set model
    in_feats = (args.hop+1)*2 
    if args.model_type == 'IGMC':
        model = IGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))
        
    # loss_fn = nn.MSELoss().to(args.device)
    loss_fn = nn.BCEWithLogitsLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    
    best_epoch = 0
    best_auc, best_acc = 0, 0

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    epochs = list(range(1, args.train_epochs+1))

    eval_func_map = {
        'IGMC': evaluate,
    }
    eval_func = eval_func_map.get(args.model_type, evaluate)
    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
    
        train_loss = train_epoch(model, loss_fn, optimizer, train_loader, 
                                 args.device, logger, args.log_interval)
        # val_rmse = eval_func(model, valid_loader, args.device)
        test_auc, test_acc = eval_func(model, test_loader, args.device)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'test_auc': test_auc,
            'test_acc': test_acc,

        }
        logger.info('=== Epoch {}, train loss {:.6f}, test auc {:.6f}, test acc {:.6f} ==='.format(*eval_info.values()))

        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_auc < test_auc:
            logger.info(f'new best test auc {test_auc:.6f} acc {test_acc:.6f} ===')
            best_epoch = epoch_idx
            best_auc = test_auc
            best_acc = test_acc
            # best_rmse = test_rmse
            # best_state = copy.deepcopy(model.state_dict())
        
    # th.save(best_state, f'./parameters/{args.key}_{args.data_name}_{best_rmse:.4f}.pt')
    logger.info(f"Training ends. The best testing auc is {best_auc:.6f} acc {best_acc:.6f} at epoch {best_epoch}")
    return test_auc
    
import yaml
from collections import defaultdict
from datetime import datetime

def main():
    while 1:
        with open('./train_configs/train_list.yaml') as f:
            files = yaml.load(f, Loader=yaml.FullLoader)
        file_list = files['files']
        for f in file_list:
            date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
            args = get_args_from_yaml(f)
            logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
            logger.info('train args')
            for k,v in args.items():
                logger.info(f'{k}: {v}')

            test_results = defaultdict(list)
            best_lr = None
            # for data_name in args.datasets:
            sub_args = args
            # sub_args['data_name'] = data_name
            best_rmse_list = []
            for lr in args.train_lrs:
                sub_args['train_lr'] = lr
                best_rmse = train(sub_args, logger=logger)
                # test_results[data_name].append(best_rmse)
                best_rmse_list.append(best_rmse)
            
            logger.info(f"**********The final best testing RMSE is {min(best_rmse_list):.6f} at lr {best_lr}********")
            logger.info(f"**********The mean testing RMSE is {np.mean(best_rmse_list):.6f}, {np.std(best_rmse_list)} ********")
        
            # mean_std_dict = dict()
            # for dataset, results in test_results.items():
            #     mean_std_dict[dataset] = [f'{np.mean(results):.4f} ± {np.std(results):.5f}']
            # mean_std_df = pd.DataFrame(mean_std_dict)
            # mean_std_df.to_csv(f'./results/{args.key}_{date_time}.csv')
        
if __name__ == '__main__':
    main()