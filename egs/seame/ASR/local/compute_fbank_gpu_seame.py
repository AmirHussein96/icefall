#!/usr/bin/env python3
# Johns Hopkins University  (authors: Amir Hussein)
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
This file computes fbank features.
It looks for manifests in the directory data_seame/manifests.

The generated fbank features are saved in data_seame/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

from lhotse import CutSet, LilcomChunkyWriter
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.recipes.utils import read_manifests_if_cached


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for the train set.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the train set split.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop index of the train set split.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, only compute features for the dev and val set.",
    )

    return parser.parse_args()


def compute_fbank_gpu(args):
    src_dir = Path("data_seame/manifests")
    output_dir = Path("data_seame/fbank")
    num_jobs = min(os.cpu_count(), 10)
    num_mel_bins = 80
    sampling_rate = 16000
    sr = 16000

    logging.info(f"Cpus {num_jobs}")

    dataset_parts = (
        "valid",
        "dev_man",
        "train",
        "dev_sge",
    )
    prefix = ""
    suffix = "jsonl.gz"
    breakpoint
    manifests = read_manifests_if_cached(
        prefix=prefix,
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix=suffix,
    )
    assert manifests is not None

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    for partition, m in manifests.items():
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / f"{cuts_filename}").is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        logging.info("About to split cuts into smaller chunks.")
        if sr != None:
            logging.info(f"Resampling to {sr}")
            cut_set = cut_set.resample(sr)

        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, keep_all_channels=False
        )
        cut_set = cut_set.filter(lambda c: c.duration >= 0.2 and c.duration <= 30)
        if "train" in partition:
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                manifest_path=f"{src_dir}/{cuts_filename}",
                batch_duration=2000,
                num_workers=num_jobs,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )
            cut_set.to_file(output_dir / f"cuts_{partition}.jsonl.gz")
        else:
            logging.info(f"Processing {partition}")
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                batch_duration=2000,
                num_workers=num_jobs,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )
            cut_set.to_file(output_dir / f"cuts_{partition}.jsonl.gz")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()

    compute_fbank_gpu(args)