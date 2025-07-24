
#!/bin/bash


data_dir=../../../train_graphs/$1

export CUDA_VISIBLE_DEVICES=0

python3 ../main.py \
  -seed 285 \
  -data_dir $data_dir \
  -graph_type $1 \
  -baseline True \
  -gpu 0 \
  $@