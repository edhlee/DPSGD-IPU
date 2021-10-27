#!/bin/bash



python train_dpsgd.py --precision 16.16  --model resnet --model-size 50 --dataset imagenet --data-dir  ~/work/imagenet-data  --internal-exchange-optimisation-target balanced \
  --ckpts-per-epoch 4 --gradient-accumulation-count 32 --batch-size 1   --replicas 16 --optimiser SGD \
  --max-cross-replica-buffer-size 100000000 --lr-schedule stepped --learning-rate-schedule 0.25,0.5,0.75,0.9  \
  --label-smoothing 0.1  --epochs 100 --enable-half-partials --normalise-input --stable-norm --learning-rate-decay 1 \
  --eight-bit-io --label-smoothing 0.1 --logs-per-epoch 10 \
  --noise-std 2.0 --clipping-threshold 4.0 




