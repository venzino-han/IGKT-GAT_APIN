# IGKT
Temporal Enhanced Inductive Graph Knowledge Tracing

# Datasets 
- EdNet dataset from : https://github.com/riiid/ednet
- ASSIST2017 dataset from : https://sites.google.com/view/assistmentsdatamining
<br />

## Download datasets
1. `make up` : build docker image and start docker container
2. `python3 download_datasets.py` : download datasets

# Docker Container
- Docker container use igkt project directory as volume 
- File change will be apply directly to file in docker container

## Preprocessing
1. `make up` : build docker image and start docker container
2. `python3 src/pre_process.py` : start ednet data preprocessing in docker container
2. `python3 src/pre_process_assist.py` : start assist data preprocessing in docker container
3. `python3 src/item_preprocessing.py` : start ednet item data preprocessing in docker container
3. `python3 src/item_preprocessing_assist.py` : start assist item data preprocessing in docker container


# Train 
1. `make up` : build docker image and start docker container
2. check `train_config/train_list.ymal` file (default: assist2017 with igkt_ts and igkt_gat)
3. `python3 src/train.py` : start train in docker container

<br />
