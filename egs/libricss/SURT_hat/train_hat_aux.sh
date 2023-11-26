#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Train pruned stateless RNN-T model
queue-freegpu.pl --gpu 4 --mem 16G --config conf/gpu.conf dprnn_zipformer_hat/exp/h2/train.log \
  python dprnn_zipformer_hat/train2.py \
    --master-port 14612 \
    --use-fp16 True \
    --exp-dir dprnn_zipformer_hat/exp/h2 \
    --world-size 4 \
    --max-duration 500 \
    --max-duration-valid 200 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 10 \
    --enable-spec-aug True \
    --enable-musan True \
    --ctc-loss-scale 0.0 \
    --heat-loss-scale 0.0 \
    --base-lr 0.004 \
    --chunk-width-randomization True \
    --model-init-ckpt dprnn_zipformer_hat/exp/h1/pretrained.pt \
    --use-aux-encoder True \
    --freeze-main-model True