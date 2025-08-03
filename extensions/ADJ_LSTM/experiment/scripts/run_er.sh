
#!/bin/bash

graph_type=er
ordering=BFS
bsize=1
hidden_dim=128
embed_dim=16
epoch_load=0
num_layers=2
weighted=True
mode=score

data_dir=../../train_graphs/$graph_type

save_dir=../../model_saves/$graph_type-weighted-$mode

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python3 ../main.py \
  -hidden_dim $hidden_dim \
  -embed_dim $embed_dim \
  -num_layers $num_layers \
  -weighted $weighted \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -graph_type $graph_type \
  -epoch_load $epoch_load \
  -epoch_save 1 \
  -batch_size $bsize \
  -mode $mode \
  -accum_grad 1 \
  -num_gen 20 \
  -num_epochs 27 \
  -gpu 0 \
  $@
