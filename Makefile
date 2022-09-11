default: build

help:
	@echo 'Management commands for kgmc:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t kgmc 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name kgmc -v `pwd`:/workspace/kgmc kgmc:latest /bin/bash

up: build run

rm: 
	@docker rm kgmc

stop:
	@docker stop kgmc

reset: stop rm

# clear:
# 	@echo "clear docker image"
# 	@docker rmi $(docker images -f "dangling=true" -q)