default: build

help:
	@echo 'Management commands for igkt:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t igkt 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name igkt -v `pwd`:/workspace/igkt igkt:latest /bin/bash

up: build run

rm: 
	@docker rm igkt

stop:
	@docker stop igkt

reset: stop rm

# clear:
# 	@echo "clear docker image"
# 	@docker rmi $(docker images -f "dangling=true" -q)