#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Decode RNNT model
queue-freegpu.pl --gpu 1 --mem 16G --config conf/gpu_decode.conf dprnn_zipformer2/exp/v5/decode_greedy.log \
  python dprnn_zipformer2/decode.py \
    --epoch 24 --avg 8 --use-averaged-model True \
    --exp-dir dprnn_zipformer2/exp/v5 \
    --max-duration 250 \
    --decoding-method greedy_search \
    --beam-size 4 \
    --use-joint-encoder-layer lstm \
    --chunk-size 32 \
    --left-context-frames 128 \
    --noise-class False \
    --linear-masking True