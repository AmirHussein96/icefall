#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/librispeech
#      You can find audio and transcripts for LibriSpeech in this path.
#
#  - $dl_dir/libricss
#      You can find audio and transcripts for LibriCSS in this path.
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#
dl_dir=$PWD/download
cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data
vocab_size=500

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/librispeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/librispeech $dl_dir/librispeech
  #
  if [ ! -d $dl_dir/librispeech ]; then
    lhotse download librispeech $dl_dir/librispeech
  fi

  # If you have pre-downloaded it to /path/to/libricss,
  # you can create a symlink
  #
  #   ln -sfv /path/to/libricss $dl_dir/libricss
  #
  if [ ! -d $dl_dir/libricss ]; then
    lhotse download libricss $dl_dir/libricss
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriSpeech manifests"
  # We assume that you have downloaded the LibriSpeech corpus
  # to $dl_dir/librispeech. We perform text normalization for the transcripts.
  # NOTE: Alignments are required for this recipe.
  mkdir -p data/manifests
  lhotse prepare librispeech -p train-clean-100 -p train-clean-360 -p train-other-500 -p dev-clean \
    -j 4 --alignments-dir $dl_dir/libri_alignments/LibriSpeech $dl_dir/librispeech data/manifests/
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare LibriCSS manifests"
  # We assume that you have downloaded the LibriCSS corpus
  # to $dl_dir/libricss. We perform text normalization for the transcripts.
  mkdir -p data/manifests
  for mic in sdm ihm-mix; do
    lhotse prepare libricss --type $mic --segmented $dl_dir/libricss data/manifests/
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to $dl_dir/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Extract features for LibriSpeech, trim to alignments, and shuffle the cuts"
  $cmd exp/extract_libri_fbank.log python local/compute_fbank_librispeech.py
  lhotse combine data/manifests/librispeech_cuts_train* - |\
    lhotse cut trim-to-alignments --type word --max-pause 0.2 - - |\
    shuf | gzip -c > data/manifests/librispeech_cuts_train_trimmed.jsonl.gz
  lhotse cut trim-to-alignments --type word --max-pause 0.2 data/manifests/librispeech_cuts_dev-clean.jsonl.gz - |\
    shuf | gzip -c > data/manifests/librispeech_cuts_dev_trimmed.jsonl.gz
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Create simulated mixtures from LibriSpeech (train and dev). This may take a while."
  # We create a 2-speaker set which will be used during the model warmup phase, and a
  # full training set (2,3,4 speakers) that will be used for the subsequent training.
  # We create anechoic and reverberant versions of both sets. For the full set, we compute
  # silence and overlap distributions based on LibriCSS sessions (no 0L).

  sim_cmd="queue.pl --mem 16G -l 'h_rt=100:00:00'"

  # gunzip -c data/manifests/libricss-sdm_supervisions_all.jsonl.gz |\
  #   grep -v "0L" | grep -v "OV10" |\
  #   gzip -c > data/manifests/libricss-sdm_supervisions_all_v1.jsonl.gz

  # gunzip -c data/manifests/libricss-sdm_supervisions_all.jsonl.gz |\
  #   grep "OV40" |\
  #   gzip -c > data/manifests/libricss-sdm_supervisions_ov40.jsonl.gz

  # Full training set (2,3 speakers) anechoic
  for part in train; do
    if [ $part == "dev" ]; then
      num_jobs=1
    else
      num_jobs=4
    fi
    log "Generating anechoic ${part} set (full)"
    $sim_cmd exp/sim_${part}_v1.log lhotse workflows simulate-meetings \
      --method conversational \
      --fit-to-supervisions data/manifests/libricss-sdm_supervisions_all_v1.jsonl.gz \
      --num-repeats 1 \
      --num-speakers-per-meeting 2,3 \
      --max-duration-per-speaker 15.0 \
      --max-utterances-per-speaker 3 \
      --seed 1234 \
      --num-jobs ${num_jobs} \
      data/manifests/librispeech_cuts_${part}_trimmed.jsonl.gz \
      data/manifests/libri-mix_cuts_${part}_norvb_v1.jsonl.gz
  done

  # Warmup mixtures (100k) based on high overlap (OV40)
  # log "Generating 100k anechoic train mixtures for warmup"
  # $sim_cmd exp/sim_train_ov40.log lhotse workflows simulate-meetings \
  #   --method conversational \
  #   --fit-to-supervisions data/manifests/libricss-sdm_supervisions_ov40.jsonl.gz \
  #   --num-meetings 100000 \
  #   --num-speakers-per-meeting 2,3 \
  #   --max-duration-per-speaker 15.0 \
  #   --max-utterances-per-speaker 3 \
  #   --seed 1234 \
  #   --num-jobs 4 \
  #   data/manifests/librispeech_cuts_train_trimmed.jsonl.gz \
  #   data/manifests/libri-mix_cuts_train_norvb_ov40.jsonl.gz
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute fbank features for musan"
  mkdir -p data/fbank
  $cmd exp/feats_musan.log python local/compute_fbank_musan.py
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compute fbank features for simulated Libri-mix"
  mkdir -p data/fbank
  $cmd exp/feats_librimix_rvb3.log python local/compute_fbank_librimix.py
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compute fbank features for LibriCSS"
  mkdir -p data/fbank
  $cmd exp/feats_libricss_wpe.log python local/compute_fbank_libricss.py
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Download LibriSpeech BPE model from HuggingFace."
  mkdir -p data/lang_bpe_500 && pushd data/lang_bpe_500
  wget https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/resolve/main/data/lang_bpe_500/bpe.model
  popd
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Add source feats to mixtures (useful for auxiliary tasks)"
  python local/add_source_feats.py
fi
