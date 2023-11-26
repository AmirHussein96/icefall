#!/usr/bin/env bash

set -eou pipefail

. parse_options.sh || exit 1

# Decode RNNT model
queue-freegpu.pl --gpu 1 --mem 16G --config conf/gpu_decode.conf dprnn_zipformer_hat/exp/h1/decode_beam.log \
  python dprnn_zipformer_hat/decode.py \
    --epoch 30 --avg 9 --use-averaged-model True \
    --exp-dir dprnn_zipformer_hat/exp/h1 \
    --max-duration 250 \
    --decoding-method modified_beam_search \
    --beam-size 4 \
    --use-joint-encoder-layer lstm \
    --chunk-size 32 \
    --left-context-frames 128