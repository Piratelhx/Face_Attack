#!/usr/bin/env sh

python -u gradient_analysis.py \
  --batch_size 1\
  --lr 0.01 \
  --model_path /media/lhx/data2T/checkpoint_of_Face_Attack/ \
  --num_classes 888 \
  --FRNet resnet18 \
  --in_ch 1 \
  --input_size 224 \
  --dset_json_path /media/lhx/lhx/depthmap_hdf5/json/ND_total_13450.json \
  --dset_hdf5_path /media/lhx/lhx/depthmap_hdf5/ND_depthmap.hdf5  \
  --restore \
  --load /media/lhx/data2T/checkpoint_of_Face_Attack/train_1_9/model_best.pth \
#   --patch
  # --train_start_index 1156
  


  
  
