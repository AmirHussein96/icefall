#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Train pruned stateless RNN-T model
queue-freegpu.pl --gpu 1 --mem 16G --config conf/gpu.conf dprnn_pruned_transducer_stateless9/exp/v10b_adapt4/train.log \
  python dprnn_pruned_transducer_stateless9/train_adapt.py \
    --use-fp16 True \
    --exp-dir dprnn_pruned_transducer_stateless9/exp/v10b_adapt4 \
    --world-size 1 \
    --max-duration 200 \
    --max-duration-valid 200 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 8 \
    --lr-epochs 2 \
    --enable-spec-aug True \
    --enable-musan False \
    --ctc-loss-scale 0.2 \
    --base-lr 0.0004 \
    --model-init-ckpt dprnn_pruned_transducer_stateless9/exp/v10b/epoch-30.pt \
    --chunk-width-randomization True \
    --use-joint-encoder-layer lstm 
    # --num-mask-encoder-layers 6 \
    # --num-encoder-layers 2,4,3,2,4 