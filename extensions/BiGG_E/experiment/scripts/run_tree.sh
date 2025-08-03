#!/bin/bash

g_type=tree
ordering=DFS
bsize=100
accum_grad=1
method=BiGG-E
data_dir=../../../train_graphs/$g_type

save_dir=../../../bigg-results/$g_type-$method

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi


python3 ../main.py \
  -seed 285 \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $ordering \
  -bits_compress 256 \
  -epoch_save 50 \
  -num_epochs 550 \
  -batch_size $bsize \
  -gpu 0 \
  -has_edge_feats 1 \
  -schedule True \
  -scale_loss 10 \
  -wt_mode score \
  -accum_grad $accum_grad \
  -method $method \
  -embed_dim_wt 16 \
  -mu_0 True \
  -wt_plateu 150 \
  -top_plateu 200 \
  -dynam_score True \
  $@




