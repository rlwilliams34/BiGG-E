#!/bin/bash

g_type=er
ordering=DFS
blksize=-1
bsize=8
accum_grad=5
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
  -bits_compress 256 \
  -epoch_save 250 \
  -num_epochs 500 \
  -bits_compress 256 \
  -batch_size $bsize \
  -gpu 0 \
  -has_edge_feats 1 \
  -schedule True \
  -scale_loss 10 \
  -wt_mode score \
  -accum_grad $accum_grad \
  -method $method \
  -embed_dim_wt 16 \
  -scale_loss 10 \
  -mu_0 True \
  -dynam_score True \
  -wt_plateu 100 \
  -top_plateu 500 \
  $@
