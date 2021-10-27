#!/bin/bash



grad_acc=16
noise_std=2.0
clip=2.5
bs=2



python train_dpsgd.py --model resnet --model-size 50 --dataset imagenet --data-dir /mnt/data/edhlee/imagenet-data \
 --shards 4 --precision 16.16 --replicas 4 --gradient-accumulation-count ${grad_acc} --epochs 100  \
 --optimiser SGD --ckpts-per-epoch 10 --enable-recomputation  \
 --available-memory-proportion 0.6 0.6 0.6 0.6 0.6 0.6 0.16 0.2 --pipeline --pipeline-splits b1/2/relu b2/3/relu b3/5/relu --pipeline-schedule Grouped \
 --max-cross-replica-buffer-size 100000000 --internal-exchange-optimisation-target balanced --normalise-input --stable-norm --lr-schedule stepped --learning-rate-schedule 0.25,0.5,0.75,0.9 \
 --enable-half-partials   --eight-bit-io  --label-smoothing 0.1 --logs-per-epoch 10 --ckpts-per-epoch 4 \
  --noise-std ${noise_std} --clipping-threshold ${clip} --batch-size ${bs} #--no-stochastic-rounding 