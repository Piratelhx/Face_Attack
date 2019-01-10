#!/usr/bin/env sh

python -u train.py \
  --batch_size 64\
  --lr 0.01 \
  --model_path /media/lhx/data2T/checkpoint_of_Face_Attack/train_1_10/ \
  --optim Adam \
  --display 50 \
  --test_iter 500 \
  --snapshot 2000 \
  --num_classes 888 \
  --FRNet resnet18 \
  --max_iter 200000000 \
  --loss AddMarginLoss \
  --step_size 1000000 \
  --in_ch 3 \
  --gamma 0.1 \
  --input_size 224 \
  --dset_json_path /media/lhx/lhx/depthmap_hdf5/json/ND_total_13450.json \
  --dset_hdf5_path /media/lhx/lhx/depthmap_hdf5/ND_depthmap.hdf5  \
  # --restore \
  # --load /media/lhx/data2T/checkpoint_of_Face_Attack/CP1156.pth \
  # --train_start_index 1156
  


  
  
