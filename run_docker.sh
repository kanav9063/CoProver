#!/bin/bash
# Launch SLIME docker container with our workspace mounted
# This gives us Megatron-LM + SGLang + all deps pre-configured

sudo docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/filesystem-m5/formal:/workspace \
  -w /workspace \
  -it slimerl/slime:latest \
  /bin/bash
