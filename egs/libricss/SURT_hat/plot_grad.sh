#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Train pruned stateless RNN-T model
queue-freegpu.pl --gpu 1 --mem 16G --config conf/gpu_decode.conf dprnn_pruned_transducer_stateless8/exp/v2/plot.log \
  python dprnn_pruned_transducer_stateless8/train.py \
    --use-fp16 True \
    --exp-dir dprnn_pruned_transducer_stateless8/exp/v2 \
    --world-size 1 \
    --max-duration 200 \
    --max-duration-valid 200 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 31 \
    --start-epoch 31 \
    --lr-epochs 6 \
    --enable-spec-aug False \
    --enable-musan True \
    --ctc-loss-scale 0.2 \
    --base-lr 0.001 \
    --chunk-width-randomization True \
    --plot-total-grad True \
    --use-joint-encoder-layer lstm 