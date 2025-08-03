#!/bin/bash

g_type=tree
ordering=DFS
bsize=20
num_leaves=50
accum_grad=1
method=BiGG-E
num_epochs=0
top_plateu=-1
wt_plateu=-1
base_path='../../../scalability_saves'

if [ ! -e $base_path ];
then
  mkdir -p $base_path
fi

python3 ../scalability.py \
  -seed 285 \
  -num_leaves $num_leaves \
  -g_type $g_type \
  -node_order $ordering \
  -batch_size $bsize \
  -method $method \
  -accum_grad $accum_grad \
  -num_epochs $num_epochs \
  -wt_plateu $top_plateu \
  -top_plateu $wt_plateu \
  -bits_compress 256 \
  -embed_dim_wt 32 \
  -schedule True \
  -scale_loss 10 \
  -mu_0 True \
  -dynam_score True \
  -wt_mode score \
  -has_edge_feats 1 \
  -gpu 0 \
  -source 0 \
  -scale_run True \
  $@

