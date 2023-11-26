#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
(1) greedy search
./dprnn_zipformer/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model true \
    --exp-dir ./dprnn_zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) modified beam search
./dprnn_zipformer/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model true \
    --exp-dir ./dprnn_zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4
"""


import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from kaldialign import edit_distance
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from asr_datamodule import LibriCssAsrDataModule
from beam_search2 import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from train2 import add_model_arguments, get_params, get_surt_model

from lhotse.utils import EPSILON
from icefall import LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
)

OVERLAP_RATIOS = ["0L", "0S", "OV10", "OV20", "OV30", "OV40"]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="dprnn_zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--save-masks",
        type=str2bool,
        default=False,
        help="""If true, save masks generated by unmixing module.""",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    feature_lens = batch["input_lens"].to(device)

    # Apply the mask encoder
    B, T, F = feature.shape
    h, h_lens, _, masks = model.forward_mask_encoder(feature, feature_lens)

    masks_dict = {}
    if params.save_masks:
        # To save the masks, we split them by batch and trim each mask to the length of
        # the corresponding feature. We save them in a dict, where the key is the
        # cut ID and the value is the mask.
        for i in range(B):
            mask = torch.cat(
                [masks[j][i, : feature_lens[i]] for j in range(params.num_channels)],
                dim=-1,
            )
            mask = mask.cpu().numpy()
            masks_dict[batch["cuts"][i].id] = mask

    # Apply the encoder
    encoder_out, encoder_out_lens, aux_encoder_out = model.forward_encoder(h, h_lens)

    N = encoder_out.size(0)
    num_channels = N // B

    if model.joint_encoder_layer is not None:
        encoder_out = model.joint_encoder_layer(encoder_out)

    if model.aux_joint_encoder_layer is not None:
        aux_encoder_out = model.aux_joint_encoder_layer(
            aux_encoder_out, encoder_out_lens
        )

    hyps = []
    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        results = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            aux_encoder_out=aux_encoder_out,
        )
        for i in range(B):
            cur_hyps = []
            for j in range(num_channels):
                cur_hyps.append(results[i + j * B])
            hyps_by_speaker = defaultdict(list)
            for hyp in cur_hyps:
                for token, spk, ts in zip(hyp.hyps, hyp.aux_hyps, hyp.timestamps):
                    hyps_by_speaker[spk].append((token, ts))
            hyps_by_speaker = dict(hyps_by_speaker)
            # For each speaker, order the tokens by timestamp. We also remove
            # duplicated tokens at the same timestamp.
            hyps_by_speaker = {
                spk: [token for token, _ in sorted(set(tokens), key=lambda x: x[1])]
                for spk, tokens in hyps_by_speaker.items()
            }
            # For each speaker, convert the tokens to words.
            hyps_by_speaker = {
                spk: sp.decode(tokens) for spk, tokens in hyps_by_speaker.items()
            }
            # pprint(hyps_by_speaker)
            hyps.append(hyps_by_speaker)

    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp)
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(sp.decode(hyp))

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}, masks_dict
    else:
        return {f"beam_size_{params.beam_size}": hyps}, masks_dict


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    masks = {}
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["cuts"]]
        cuts_batch = batch["cuts"]

        hyps_dict, masks_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )
        masks.update(masks_dict)

        for name, hyps in hyps_dict.items():
            this_batch = []
            for cut_id, hyp_words in zip(cut_ids, hyps):
                # Reference is a list of supervision texts sorted by start time.
                # Group reference supervisions by speaker.
                ref_words = defaultdict(list)
                for s in sorted(cuts_batch[cut_id].supervisions, key=lambda s: s.start):
                    ref_words[s.speaker].append(s.text.strip())
                ref_words = dict(ref_words)
                # Convert reference words to a single string.
                ref_words = {spk: " ".join(words) for spk, words in ref_words.items()}
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results, masks_dict


def compute_cpWER(ref_text, hyp_text):
    """
    ref_text and hyp_text are lists of strings.
    """
    M = len(ref_text)
    N = len(hyp_text)
    costs = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cur_ref = ref_text[i].split()
            cur_hyp = hyp_text[j].split()
            result = edit_distance(cur_ref, cur_hyp)
            wer = result["total"] / len(cur_ref)
            costs[i, j] = wer
    row_ind, col_ind = linear_sum_assignment(costs)
    ref_text = [ref_text[i] for i in row_ind]
    hyp_text = [hyp_text[i] for i in col_ind]
    count = num_ins = num_del = num_sub = total = 0
    for ref, hyp in zip(ref_text, hyp_text):
        ref = ref.strip().split()
        hyp = hyp.strip().split()
        count += len(ref)
        result = edit_distance(ref, hyp)
        num_ins += result["ins"]
        num_del += result["del"]
        num_sub += result["sub"]
        total += result["total"]
    return {
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "count": count,
        "num_ins": num_ins,
        "num_del": num_del,
        "num_sub": num_sub,
        "ins": num_ins / count,
        "del": num_del / count,
        "sub": num_sub / count,
        "cpwer": total / count,
    }


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        stats_path = params.res_dir / f"stats-{test_set_name}-key-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        wer_dict = {}
        num_spk_dict = {}
        # Each cut in the reference corresponds to a recording, so we iterate over all the
        # cuts (i.e. recordings)
        for reco_id, ref_words, hyp_words in results:

            ref_text = list(ref_words.values())
            hyp_text = list(hyp_words.values())

            # Compute cpWER
            stats = compute_cpWER(ref_text, hyp_text)

            # Store results
            wer_dict[reco_id] = stats
            num_spk_dict[reco_id] = (len(ref_text), len(hyp_text))

        # Compute average cpWER
        total_num_words = sum(wer_dict[reco_id]["count"] for reco_id in wer_dict)
        total_ins = sum(wer_dict[reco_id]["num_ins"] for reco_id in wer_dict)
        total_del = sum(wer_dict[reco_id]["num_del"] for reco_id in wer_dict)
        total_sub = sum(wer_dict[reco_id]["num_sub"] for reco_id in wer_dict)
        avg_ins = total_ins / total_num_words
        avg_del = total_del / total_num_words
        avg_sub = total_sub / total_num_words
        avg_cpwer = (total_ins + total_del + total_sub) / total_num_words

        wer_dict["TOTAL"] = {
            "ref_text": [],
            "hyp_text": [],
            "count": total_num_words,
            "num_ins": total_ins,
            "num_del": total_del,
            "num_sub": total_sub,
            "ins": avg_ins,
            "del": avg_del,
            "sub": avg_sub,
            "cpwer": avg_cpwer,
        }

        # Write results to file
        with stats_path.open("w") as f:
            json.dump(wer_dict, f, indent=2)

        # Print averages
        print(f"Average insertion rate: {avg_ins:.2%}")
        print(f"Average deletion rate: {avg_del:.2%}")
        print(f"Average substitution rate: {avg_sub:.2%}")
        print(f"Average cpWER: {avg_cpwer:.2%}")

        # Print confusion matrix of number of speakers
        y_true = [x[0] for x in num_spk_dict.values()]
        y_pred = [x[1] for x in num_spk_dict.values()]
        cm = confusion_matrix(y_true, y_pred, labels=range(1, max(y_true) + 1))
        print("Confusion matrix of number of speakers:")
        print(cm)


def save_masks(
    params: AttributeDict,
    test_set_name: str,
    masks: List[torch.Tensor],
):
    masks_path = params.res_dir / f"masks-{test_set_name}.pt"
    torch.save(masks, masks_path)
    logging.info(f"The masks are stored in {masks_path}")


@torch.no_grad()
def main():
    parser = get_parser()
    LmScorer.add_arguments(parser)
    LibriCssAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "modified_beam_search",
    ), f"Decoding method {params.decoding_method} is not supported."
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    assert "," not in params.chunk_size, "chunk_size should be one value in decoding."
    assert (
        "," not in params.left_context_frames
    ), "left_context_frames should be one value in decoding."
    params.suffix += f"-chunk-{params.chunk_size}"
    params.suffix += f"-left-context-{params.left_context_frames}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_surt_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    libricss = LibriCssAsrDataModule(args)

    # dev_cuts = libricss.libricss_cuts(split="dev", type="ihm-mix").to_eager()
    # dev_cuts_grouped = [dev_cuts.filter(lambda x: ol in x.id) for ol in OVERLAP_RATIOS]
    # test_cuts = libricss.libricss_cuts(split="test", type="ihm-mix").to_eager()
    # test_cuts_grouped = [
    #     test_cuts.filter(lambda x: ol in x.id) for ol in OVERLAP_RATIOS
    # ]

    from lhotse import load_manifest_lazy

    dev_cuts = load_manifest_lazy("data/manifests/cuts_dev_norvb_v1_sources.jsonl.gz")
    dev_dl = libricss.test_dataloaders(dev_cuts, text_delimiter="|")
    results_dict, masks = decode_dataset(
        dl=dev_dl,
        params=params,
        model=model,
        sp=sp,
    )
    save_results(
        params=params,
        test_set_name=f"dev",
        results_dict=results_dict,
    )

    # for dev_set, ol in zip(dev_cuts_grouped, OVERLAP_RATIOS):
    #     dev_dl = libricss.test_dataloaders(dev_set, text_delimiter="|")
    #     results_dict, masks = decode_dataset(
    #         dl=dev_dl,
    #         params=params,
    #         model=model,
    #         sp=sp,
    #     )

    #     save_results(
    #         params=params,
    #         test_set_name=f"dev_{ol}",
    #         results_dict=results_dict,
    #     )

    #     if params.save_masks:
    #         save_masks(
    #             params=params,
    #             test_set_name=f"dev_{ol}",
    #             masks=masks,
    #         )

    # for test_set, ol in zip(test_cuts_grouped, OVERLAP_RATIOS):
    #     test_dl = libricss.test_dataloaders(test_set, text_delimiter="|")
    #     results_dict, masks = decode_dataset(
    #         dl=test_dl,
    #         params=params,
    #         model=model,
    #         sp=sp,
    #     )

    #     save_results(
    #         params=params,
    #         test_set_name=f"test_{ol}",
    #         results_dict=results_dict,
    #     )

    #     if params.save_masks:
    #         save_masks(
    #             params=params,
    #             test_set_name=f"test_{ol}",
    #             masks=masks,
    #         )

    logging.info("Done!")


if __name__ == "__main__":
    main()
