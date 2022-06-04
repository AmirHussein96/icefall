#!/usr/bin/env bash
# Copyright 2022 Johns Hopkins University  (Amir Hussein)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail
nj=30
stage=7
stop_stage=1000

# We assume dl_dir (download dir) contains the following
# directories and files. 
#
#  - $dl_dir/mgb2
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
# Note: MGB2 is not available for direct 
# download, however you can fill out the form and  
# download it from https://arabicspeech.org/mgb2 

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  5000
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
  #   ln -sfv /path/to/mgb2 $dl_dir/MGB2

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
  log "Stage 1: Prepare mgb2 manifest"
  # We assume that you have downloaded the mgb2 corpus
  # to $dl_dir/mgb2
  mkdir -p data/manifests

  lhotse prepare mgb2 $dl_dir/mgb2 data/manifests
  
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for mgb2"
  mkdir -p data/fbank
  ./local/compute_fbank_mgb2.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  if [[ ! -e download/lm/train/text ]]; then 
  # export train text file to build grapheme lexicon 
  lhotse kaldi export \
    data/manifests/mgb2_recordings_train.jsonl.gz \
    data/manifests/mgb2_supervisions_train.jsonl.gz  \
    download/lm/train
  fi

  lang_dir=data/lang_phone
  mkdir -p $lang_dir
  ./local/prep_lexicon.sh 
  python local/prepare_lexicon.py  $dl_dir/lm/grapheme_lexicon.txt  $dl_dir/lm/lexicon.txt
  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $dl_dir/lm/lexicon.txt |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      files=$(
        find "$dl_dir/lm/train" -name "text"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2- | sed -r '/^\s*$/d'
      done > $lang_dir/transcript_words.txt
    fi

    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --vocab-size $vocab_size \
      --transcript $lang_dir/transcript_words.txt

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir
    fi
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare bigram P"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/transcript_tokens.txt ]; then
      ./local/convert_transcript_words_to_tokens.py \
        --lexicon $lang_dir/lexicon.txt \
        --transcript $lang_dir/transcript_words.txt \
        --oov "<UNK>" \
        > $lang_dir/transcript_tokens.txt
    fi

    if [ ! -f $lang_dir/P.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 2 \
        -text $lang_dir/transcript_tokens.txt \
        -lm $lang_dir/P.arpa
    fi

    if [ ! -f $lang_dir/P.fst.txt ]; then
      python3 -m kaldilm \
        --read-symbol-table="$lang_dir/tokens.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        $lang_dir/P.arpa > $lang_dir/P.fst.txt
    fi
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm
  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p data/lm
    if [ ! -f data/lm/G_3_gram.fst.txt ]; then
      # It is used in building HLG
      ./shared/make_kn_lm.py \
          -ngram-order 3 \
          -text $lang_dir/transcript_words.txt \
          -lm $lang_dir/G.arpa

      python3 -m kaldilm \
        --read-symbol-table="data/lang_phone/words.txt" \
        --disambig-symbol='#0' \
        --max-order=3 \
        $lang_dir/G.arpa > data/lm/G_3_gram.fst.txt
    fi

    if [ ! -f data/lm/G_4_gram.fst.txt ]; then
      # It is used for LM rescoring
      ./shared/make_kn_lm.py \
          -ngram-order 4 \
          -text $lang_dir/transcript_words.txt \
          -lm $lang_dir/4-gram.arpa

      python3 -m kaldilm \
        --read-symbol-table="data/lang_phone/words.txt" \
        --disambig-symbol='#0' \
        --max-order=4 \
        $lang_dir/4-gram.arpa > data/lm/G_4_gram.fst.txt
    fi
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir
  done
fi
