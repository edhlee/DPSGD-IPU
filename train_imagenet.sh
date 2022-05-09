#!/bin/bash


POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout":"1200"}' poprun -v --numa-aware 1 --num-instances 8 --num-replicas 16 --ipus-per-replica 1 --mpi-local-args="--tag-output" \
python3 train_dpsgd.py --precision 16.32  --model resnet --model-size 50 --dataset imagenet --data-dir  ~/work/imagenet-data  --internal-exchange-optimisation-target balanced \
  --ckpts-per-epoch 4 --gradient-accumulation-count 512 --batch-size 1   --replicas 16 --optimiser DPSGD \
  --max-cross-replica-buffer-size 100000000 --lr-schedule stepped --learning-rate-schedule 0.25,0.5,0.75,0.9  \
  --label-smoothing 0.1  --epochs 300 --enable-half-partials --normalise-input --stable-norm --learning-rate-decay 0.1 \
  --eight-bit-io --label-smoothing 0.1 --logs-per-epoch 10 \
  --noise-mult 0.3  --clipping-threshold 30.0 --loss-scaling 1.0 --abs-learning-rate 1.0
