
#!/bin/bash

g_type=tree
node_order=BFS

save_dir=../../train_graphs/$g_type

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0


python3 ../run_data_creator.py \
  -weighted True \
  -save_dir $save_dir \
  -g_type $g_type \
  -node_order $node_order \
  -gpu 0 \
  -num_graphs 1000 \
  -num_leaves 100 \
  -source 0 \
  $@


