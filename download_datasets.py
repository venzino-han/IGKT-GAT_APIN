import gdown
import argparse
from itertools import chain
import os 

assist_files_dict ={
    'assist/assist.csv' : '1YWuiE2wYhepN7P6Jo51mproW7_sEANrO',
    'assist/questions.csv' : '1JaZDZC0JmOqdS5g1WJt24_sZDt_art-N',
}

ednet_files_dict = {
    'ednet/train_30m.csv' : '1XlTBPBFYEzzy4dUhAYFC78mKGnXmz9X_',
    'ednet/questions.csv' : '1drZV1NkJDuufGkIUDJdeeyW7O61XnWkm',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default="all")
    parser.add_argument("-p","--path", type=str, default="./data")
    args = parser.parse_args()

    for p in ['assist', 'ednet', 'junyi']:
        path = f'{args.path}/{p}'
        os.makedirs(path, exist_ok=True)
    
    if args.dataset == 'all':
        dataset_files_dict = dict(chain(assist_files_dict.items(), ednet_files_dict.items()))
    if args.dataset == 'assist':
        dataset_files_dict = assist_files_dict
    if args.dataset == 'ednet':
        dataset_files_dict = ednet_files_dict
    for output, url  in dataset_files_dict.items():
        gdown.download(f'https://drive.google.com/uc?id={url}', f'{args.path}/{output}', quiet=False)