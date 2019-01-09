#!/usr/bin/env sh

python -u train.py \
  --batch_size 32\
  --lr 0.01 \
  --model_path ./log \
  --optim Adam \
  --display 50 \
  --snapshot 500 \
  --num_classes 8 \
  --FRNet resnet18 \
  --max_iter 200000 \
  --loss AddMarginLoss \
  --step_size 40000 \
  --in_ch 3 \
  --input_size 256 \
  --dset_json_path /home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.json \
  --dset_hdf5_path /home/lhx/xiaozl/test_ND/ND_total_depthmap_13450.hdf5 \
  


  
  
