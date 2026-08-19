#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Master split generator
===============================================================================

Author
------
Laura Pérez Liens

Description
-----------
Generate a train / validation / test split for every dataset under
data_input_nn.

The split is performed using group_id.

Each group_id is considered an independent sample and is assigned to one
of the following datasets:

    TRAIN
    VALIDATION
    TEST

The output is a new file:

    master_split.csv

which is identical to master_index.csv but contains an additional column:

    split_id

where

    0 -> train
    1 -> validation
    2 -> test

The script automatically discovers every embedding dataset.

===============================================================================
"""

from pathlib import Path
from collections import defaultdict
import argparse


import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

TRAIN = 0
VALIDATION = 1
TEST = 2

CHUNK_SIZE = 100000

# =============================================================================
# Dataset discovery
# =============================================================================

def discover_datasets(root):
    """
    Discover every master_index.csv under data_input_nn.
    """

    root = Path(root)

    datasets = []

    for path in sorted(root.rglob("master_index.csv")):

        datasets.append(path)

    return datasets

# =============================================================================
# Statistics
# =============================================================================

class DatasetStats:

    def __init__(self):

        self.windows = defaultdict(int)

        self.positive = defaultdict(int)

        self.group_sets = {
            TRAIN: set(),
            VALIDATION: set(),
            TEST: set()
        }
# =============================================================================
# Printing
# =============================================================================

def print_header(title):

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)

def print_split_statistics(stats):

    print()

    print("Windows")

    print("-----------------------------")

    for split in [TRAIN, VALIDATION, TEST]:

        print(
            f"{split}: "
            f"{stats.windows[split]:,}"
        )

    print()

    print("Positive windows")

    print("-----------------------------")

    for split in [TRAIN, VALIDATION, TEST]:

        total = stats.windows[split]

        positive = stats.positive[split]

        ratio = positive / total if total else 0

        print(
            f"{split}: "
            f"{positive:,} "
            f"({ratio:.2%})"
        )

    print()

    print("Groups")

    print("-----------------------------")

    for split in [TRAIN, VALIDATION, TEST]:

        print(
            f"{split}: "
            f"{len(stats.group_sets[split]):,}"
        )

# =============================================================================
# Group statistics
# =============================================================================

def compute_group_statistics(master_index):
    """
    Compute the label associated with every group_id.

    Returns
    -------
    group_labels : dict

        group_id -> label
    """

    group_labels = {}

    total_rows = 0

    reader = pd.read_csv(
        master_index,
        usecols=["group_id", "y"],
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        print(
            f"\rReading chunk {chunk_number}",
            end="",
            flush=True
        )

        grouped = (
            chunk
            .groupby("group_id")["y"]
            .max()
        )

        for group_id, label in grouped.items():

            group_id = int(group_id)
            label = int(label)

            if group_id in group_labels:

                if group_labels[group_id] != label:

                    raise RuntimeError(
                        f"group_id {group_id} has inconsistent labels."
                    )

            else:

                group_labels[group_id] = label

        total_rows += len(chunk)

    print()

    print(f"Total windows : {total_rows:,}")
    print(f"Total groups  : {len(group_labels):,}")

    return group_labels

# =============================================================================
# Positive / negative groups
# =============================================================================

def split_groups(group_labels):
    """
    Split group_ids according to their class.
    """

    positive = []
    negative = []

    for group_id, label in group_labels.items():

        if label == 1:

            positive.append(group_id)

        else:

            negative.append(group_id)

    print(f"Positive groups : {len(positive):,}")
    print(f"Negative groups : {len(negative):,}")

    return positive, negative

# =============================================================================
# Stratified split
# =============================================================================

def assign_split(ids,
                 split_id,
                 split_map):

    for group_id in ids:

        split_map[group_id] = split_id


def stratified_group_split(positive_groups,
                           negative_groups,
                           train_ratio,
                           validation_ratio,
                           test_ratio,
                           seed):
    """
    Stratified split using group_id.
    """

    rng = np.random.default_rng(seed)

    split_map = {}

    for groups in [positive_groups, negative_groups]:

        groups = np.asarray(groups)

        rng.shuffle(groups)

        n = len(groups)

        n_train = int(round(n * train_ratio))

        n_validation = int(round(n * validation_ratio))

        train = groups[:n_train]

        validation = groups[
            n_train:
            n_train + n_validation
        ]

        test = groups[
            n_train + n_validation:
        ]

        assign_split(
            train,
            TRAIN,
            split_map
        )

        assign_split(
            validation,
            VALIDATION,
            split_map
        )

        assign_split(
            test,
            TEST,
            split_map
        )

    return split_map

# =============================================================================
# Validation
# =============================================================================

def validate_split(split_map,
                   group_labels):
    """
    Validate generated split.
    """

    if len(split_map) != len(group_labels):

        raise RuntimeError(
            "Some groups were not assigned."
        )

    counts = defaultdict(int)

    for split in split_map.values():

        counts[split] += 1

    print()

    print("Groups")

    print("-----------------------------")

    print(f"Train      : {counts[TRAIN]:,}")

    print(f"Validation : {counts[VALIDATION]:,}")

    print(f"Test       : {counts[TEST]:,}")

    print()

    return counts

# =============================================================================
# Master split writer
# =============================================================================

def write_master_split(master_index,
                       split_map):
    """
    Create master_split.csv.

    The input file is read chunk-by-chunk and a new column
    (split_id) is appended.

    Statistics are computed during the writing process.
    """

    output = master_index.parent / "master_split.csv"

    stats = DatasetStats()

    first_chunk = True

    reader = pd.read_csv(
        master_index,
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        print(
            f"\rWriting chunk {chunk_number}",
            end="",
            flush=True
        )

        chunk["split_id"] = chunk["group_id"].map(split_map)

        if chunk["split_id"].isna().any():

            raise RuntimeError(
                "Some group_id values have not been assigned."
            )

        chunk["split_id"] = chunk["split_id"].astype(np.uint8)


        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        for split in [TRAIN, VALIDATION, TEST]:

            subset = chunk[
                chunk["split_id"] == split
            ]

            stats.windows[split] += len(subset)

            stats.positive[split] += int(
                subset["y"].sum()
            )

            stats.group_sets[split].update(
                subset["group_id"].tolist()
            )

        # ----------------------------------------------------------
        # Write chunk
        # ----------------------------------------------------------

        chunk.to_csv(
            output,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False

    print()

    return output, stats


# =============================================================================
# Output statistics
# =============================================================================

def validate_output(stats):

    total = sum(stats.windows.values())

    print()

    print("Window distribution")
    print("-----------------------------")

    for split in [TRAIN, VALIDATION, TEST]:

        windows = stats.windows[split]

        ratio = windows / total if total else 0

        print(
            f"{split}: "
            f"{windows:,} "
            f"({ratio:.2%})"
        )

    print_split_statistics(stats)

# =============================================================================
# Dataset processing
# =============================================================================

def process_dataset(master_index,
                    train_ratio,
                    validation_ratio,
                    test_ratio,
                    seed):

    print_header(master_index.parent)

    group_labels = compute_group_statistics(
        master_index
    )

    positive, negative = split_groups(
        group_labels
    )

    split_map = stratified_group_split(
        positive,
        negative,
        train_ratio,
        validation_ratio,
        test_ratio,
        seed
    )

    validate_split(
        split_map,
        group_labels
    )

    output, stats = write_master_split(
        master_index,
        split_map
    )

    if sum(stats.windows.values()) == 0:

        raise RuntimeError(
        "No windows written."
        )

    validate_output(stats)

    print()

    print(f"Saved: {output}")

    print()

# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate master_split.csv for every embedding dataset."
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing data_input_nn."
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.80,
        help="Training ratio."
    )

    parser.add_argument(
        "--validation",
        type=float,
        default=0.10,
        help="Validation ratio."
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.10,
        help="Test ratio."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    if not np.isclose(
        args.train +
        args.validation +
        args.test,
        1.0
    ):

        raise RuntimeError(
            "Split ratios must sum to 1."
        )

    datasets = discover_datasets(
        args.root
    )

    if len(datasets) == 0:

        raise RuntimeError(
            "No datasets found."
        )

    print()

    print("=" * 80)

    print(
        f"Datasets found: {len(datasets)}"
    )

    print("=" * 80)

    for master_index in datasets:

        process_dataset(
            master_index,
            args.train,
            args.validation,
            args.test,
            args.seed
        )

    print()

    print("=" * 80)

    print("Finished successfully.")

    print("=" * 80)

    print()

if __name__ == "__main__":

    main()
