# igkt
inductive graph knowledge tracing model

# Datasets 
- EdNet dataset from : https://github.com/riiid/ednet
- ASSIST2017 dataset from : https://sites.google.com/view/assistmentsdatamining
- Pre-processed dataset in `data` directory. (only ASSIST2017 dataset in this repository)
<br />

# Docker Container
- Docker container use cgmc project directory as volume 
- File change will be apply directly to file in docker container

# Train 
1. `make up` : build docker image and start docker container
2. check `train_config/train_list.ymal` file (default: assist2017 with igmc, igkt, igkt_ts and igkt_gat)
3. `python3 src/train.py` : start train in docker container

<br />
