#!/usr/bin/env bash
# Copyright 2023 Johns Hopkins University  (Amir Hussein)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

nj=20
stage=1
stop_stage=4

# We assume dl_dir (download dir) contains the following
# directories and files.
#
#  - $dl_dir/iwslt_ta
#
#      You can download the data from
#
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#
# Note: iwslt_ta is not available for direct
# download, "Download IWSLT Tunisian from LDC LDC2022E01. This script assumes you prepared the stm files"
#"Check the instructions to prepare the stm files from the raw data here https://github.com/kevinduh/iwslt22-dialect"

dl_dir=$PWD/download
. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  1000
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/MGB2,
  # you can create a symlink
  #
  #   ln -sfv /path/to/iwslt_ta $dl_dir/iwslt_ta

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi
fbank=data/fbank
manifests=data/manifests
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare iwslt manifest"
  # We assume that you have downloaded the mgb2 corpus
  # to $dl_dir/mgb2
  manifests=data/manifests
  mkdir -p $manifests
  python /alt-arabic/speech/amir/competitions/IWSLT/lhotse/lhotse/recipes/iwslt.py

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p $manifests
  lhotse prepare musan $dl_dir/musan $manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank features"
  mkdir -p ${fbank}
  python local/compute_fbank_gpu.py --num-splits 20

  log "Combine features from train splits (may take ~1h)"
  if [ ! -f $manifests/cuts_train.jsonl.gz ]; then
    pieces=$(find $manifests -name "cuts_train_[0-9]*.jsonl.gz")
    lhotse combine $pieces $manifests/cuts_train.jsonl.gz
  fi
  gunzip -c $manifests/cuts_train.jsonl.gz | shuf | gzip -c > ${fbank}/cuts_train_shuf.jsonl.gz
fi



if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p ${fbank}
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir_src=data/lang_phone_src
  lang_dir_tgt=data/lang_phone_tgt

  if [ ! -f download/lm/train_src/transcript_words.txt ] || [ ! -f download/lm/train_tgt/transcript_words.txt ]; then
  # export train text file to build grapheme lexicon
    log "Creating transcripts in download/lm/train from lhotse cuts"
    mkdir -p download/lm/train_src
    mkdir -p download/lm/train_tgt
    python local/prepare_transcripts.py --cut ${fbank}/cuts_train_shuf.jsonl.gz --src-langdir download/lm/train_src --tgt-langdir download/lm/train_tgt
  fi
  mkdir -p $lang_dir_src
  mkdir -p $lang_dir_tgt

  log "Prepare lexicon"

  ./local/prep_lexicon.sh download/lm/train_src download/lm/train_tgt
  python local/prepare_lexicon.py  $dl_dir/lm/train_src/words.txt  $dl_dir/lm/train_src/lexicon.txt
  python local/prepare_lexicon.py  $dl_dir/lm/train_tgt/words.txt  $dl_dir/lm/train_tgt/lexicon.txt

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/train_src/lexicon.txt |
    sort | uniq > $lang_dir_src/lexicon.txt

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/train_tgt/lexicon.txt |
    sort | uniq > $lang_dir_tgt/lexicon.txt

  if [ ! -f $lang_dir_src/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir_src
  fi

  if [ ! -f $lang_dir_tgt/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir_tgt
  fi

fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"
  srctag=ta
  tgttag=en
  for vocab_size in ${vocab_sizes[@]}; do
    src_lang_dir=data/lang_bpe_${srctag}_${vocab_size}
    tgt_lang_dir=data/lang_bpe_${tgttag}_${vocab_size}
    mkdir -p ${src_lang_dir}
    mkdir -p ${tgt_lang_dir}
      # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone_src/words.txt $src_lang_dir
    cp data/lang_phone_tgt/words.txt $tgt_lang_dir
    if [ ! -f $src_lang_dir/transcript_words.txt ] || [ ! -f $tgt_lang_dir/transcript_words.txt ]; then
      log "Generate data for ${srctag} and ${tgttag} BPE training from data/fbank/cuts_train_shuf.jsonl.gz"
      python local/prepare_transcripts.py --cut ${fbank}/cuts_train_shuf.jsonl.gz --src-langdir ${src_lang_dir} --tgt-langdir ${tgt_lang_dir}
    fi
    for lang_dir in $src_lang_dir $tgt_lang_dir; do
    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript $lang_dir/transcript_words.txt

      if [ ! -f $lang_dir/L_disambig.pt ]; then
        ./local/prepare_lang_bpe.py --lang-dir $lang_dir
      fi
    done
  done
fi
