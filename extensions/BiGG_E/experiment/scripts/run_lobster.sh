#!/bin/bash

g_type=lobster
ordering=DFS
blksize=-1
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
  -data_dir $data_dir \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $ordering \
  -method $method \
  -epoch_save 50 \
  -bits_compress 256 \
  -batch_size $bsize \
  -num_epochs 250 \
  -epoch_plateu 50 \
  -gpu 0 \
  -has_edge_feats 1 \
  -schedule True \
  -scale_loss 10 \
  -wt_scale 10 \
  -wt_mode score \
  -accum_grad $accum_grad \
  -embed_dim_wt 16 \
  -wt_plateu 50 \
  -top_plateu 100 \
  -mu_0 True \
  -dynam_score True \
  $@