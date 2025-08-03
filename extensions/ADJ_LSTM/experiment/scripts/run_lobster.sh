
#!/bin/bash

graph_type=lobster
ordering=DFS
bsize=100
hidden_dim=128
embed_dim=16
epoch_load=0
num_layers=2
weighted=True
wt_mode=score

data_dir=../../../train_graphs/$graph_type


save_dir=../../model_saves/$graph_type

if [ ! -e $save_dir ];
then
  mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0
  
python3 ../main.py \
  -seed 285 \
  -hidden_dim $hidden_dim \
  -embed_dim $embed_dim \
  -num_layers $num_layers \
  -weighted $weighted \
  -data_dir $data_dir \
  -save_dir $save_dir \
  -graph_type $graph_type \
  -epoch_load $epoch_load \
  -epoch_save 100 \
  -batch_size $bsize \
  -wt_mode $wt_mode \
  -num_gen 200 \
  -num_epochs 300 \
  -gpu 0 \
  $@