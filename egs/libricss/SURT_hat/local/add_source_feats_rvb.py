#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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
This file adds source features as temporal arrays to the mixture manifests.
It looks for manifests in the directory data/manifests.
"""
import argparse
import logging
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from lhotse import (
    CutSet,
    load_manifest_lazy,
)
from lhotse.lazy import LazySlicer


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add source features to the mixture manifests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    return parser


def apply_fn(mixed_cuts_rvb, mixed_cuts_norvb, manifest_path, progress_bar=True):
    if progress_bar:
        progress = tqdm(
            total=len(mixed_cuts_rvb),
            desc="Adding source features to the mixed cuts",
        )
    else:
        progress = None
    with CutSet.open_writer(manifest_path) as cut_writer:
        for cut_rvb, cut_norvb in zip(mixed_cuts_rvb, mixed_cuts_norvb):
            assert cut_rvb.id == cut_norvb.id + "_rvb"
            cut_rvb.source_feats = cut_norvb.source_feats
            cut_rvb.source_feat_offsets = cut_norvb.source_feat_offsets
            cut_writer.write(cut_rvb)
            if progress is not None:
                progress.update(1)
    return


def add_source_feats(num_jobs=1):
    src_dir = Path("data/manifests")

    mixed_name_rvb = "train_rvb3_ov40"
    mixed_name_norvb = "train_norvb_ov40_sources"

    logging.info("Reading manifests")

    logging.info("Reading mixed cuts")
    mixed_cuts_rvb = load_manifest_lazy(src_dir / f"cuts_{mixed_name_rvb}.jsonl.gz")
    mixed_cuts_norvb = load_manifest_lazy(src_dir / f"cuts_{mixed_name_norvb}.jsonl.gz")

    logging.info("Adding source features to the reverberated mixed cuts")
    if num_jobs == 1:
        apply_fn(
            mixed_cuts_rvb,
            mixed_cuts_norvb,
            src_dir / f"cuts_{mixed_name_rvb}_sources.jsonl.gz",
            progress_bar=True,
        )
    else:
        cut_sets_rvb = [
            CutSet(LazySlicer(mixed_cuts_rvb, k=i, n=num_jobs)) for i in range(num_jobs)
        ]
        cut_sets_norvb = [
            CutSet(LazySlicer(mixed_cuts_norvb, k=i, n=num_jobs))
            for i in range(num_jobs)
        ]
        executor = ProcessPoolExecutor(
            num_jobs, mp_context=multiprocessing.get_context("spawn")
        )

        # Submit the chunked tasks to parallel workers.
        # Each worker runs the non-parallel version of this function inside.
        futures = [
            executor.submit(
                apply_fn,
                cs_rvb,
                cs_norvb,
                manifest_path=src_dir / f"cuts_{mixed_name_rvb}_sources_{i}.jsonl.gz",
                progress_bar=False,
            )
            for i, cs_rvb, cs_norvb in zip(
                range(num_jobs), cut_sets_rvb, cut_sets_norvb
            )
        ]

        # Wait for all the workers to finish.
        for future in tqdm(
            futures, desc="Waiting for the workers to finish", total=len(futures)
        ):
            future.result()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_parser().parse_args()
    add_source_feats(num_jobs=args.num_jobs)
