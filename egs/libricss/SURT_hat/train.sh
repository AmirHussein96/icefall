#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Train pruned stateless RNN-T model
queue-freegpu.pl --gpu 4 --mem 16G --config conf/gpu_v100.conf dprnn_zipformer2/exp/v5/train.log \
  python dprnn_zipformer2/train.py \
    --master-port 14666 \
    --use-fp16 True \
    --exp-dir dprnn_zipformer2/exp/v5 \
    --world-size 4 \
    --max-duration 650 \
    --max-duration-valid 200 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 30 \
    --enable-spec-aug True \
    --enable-musan True \
    --ctc-loss-scale 0.2 \
    --heat-loss-scale 0.2 \
    --base-lr 0.004 \
    --chunk-width-randomization True \
    --noise-class False \
    --linear-masking True \
    --model-init-ckpt exp/zipformer_1spk_epoch10.pt