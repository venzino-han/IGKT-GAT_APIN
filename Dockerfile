FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Change root user to use 'apt-get'
# USER root 
# RUN sudo apt-get update && \
# apt-get install -y libpq-dev libmysqlclient-dev gcc build-essential

# pip install 
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install -r requirements.txt
WORKDIR /igkt