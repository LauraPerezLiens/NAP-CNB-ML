#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Master split generator - GROUP-AWARE, memory-efficient version
===============================================================================

Generate train / validation / test splits using group_id as the indivisible
splitting unit.

CRITICAL RULE
-------------
Every row sharing the same group_id is assigned to exactly ONE split.

Therefore:

    group_id = X
        row 1  -> train
        row 2  -> train
        row 3  -> train
        ...

and NEVER:

    group_id = X
        row 1  -> train
        row 2  -> validation
        row 3  -> test

protein_group_id is NOT used to define the split. Different group_id values
belonging to the same protein_group_id are allowed to occur in different
splits.

Strategy
--------
PASS 1:
    Stream master_index.csv and discover unique group_id values and their
    class labels.

    A group_id is required to have one unique y value.

    Positive and negative groups are then independently shuffled and split
    approximately 80 / 10 / 10.

PASS 2:
    Stream master_index.csv again.

    For every row:
        group_id -> split_id

    Write master_split.csv incrementally.

This guarantees:
    - no group_id leakage
    - reproducibility with a fixed seed
    - stratification at GROUP level by y
    - low RAM usage relative to the total number of rows
    - no dependency on physical chunk boundaries

Output:
    master_split.csv

with:
    split_id

where:
    0 -> train
    1 -> validation
    2 -> test
===============================================================================
"""

from pathlib import Path
from collections import defaultdict

import argparse
import os
import threading
import time

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None


# =============================================================================
# Constants
# =============================================================================

TRAIN = 0
VALIDATION = 1
TEST = 2

UNASSIGNED = np.uint8(255)


# =============================================================================
# Dataset discovery
# =============================================================================

def discover_datasets(root):
    """Discover every master_index.csv under root."""

    root = Path(root)

    return sorted(
        root.rglob("master_index.csv")
    )


# =============================================================================
# Printing
# =============================================================================

def print_header(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def split_name(split):

    names = {
        TRAIN: "Train",
        VALIDATION: "Validation",
        TEST: "Test",
    }

    return names[split]


# =============================================================================
# System monitor
# =============================================================================

def monitor_system(interval):

    if psutil is None:

        print(
            "WARNING: psutil is not installed. "
            "System monitoring is disabled.",
            flush=True,
        )

        return

    process = psutil.Process(
        os.getpid()
    )

    psutil.cpu_percent(
        interval=None
    )

    while True:

        try:

            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            process_memory = (
                process.memory_info().rss
                / 1024**3
            )

            print(
                "\n[SYSTEM] "
                f"RAM={memory.used / 1024**3:.1f}/"
                f"{memory.total / 1024**3:.1f} GB "
                f"({memory.percent:.1f}%) | "
                f"Process={process_memory:.2f} GB | "
                f"Swap={swap.used / 1024**3:.1f}/"
                f"{swap.total / 1024**3:.1f} GB | "
                f"CPU={psutil.cpu_percent(interval=None):.1f}%",
                flush=True,
            )

        except Exception as exc:

            print(
                f"\n[SYSTEM] Monitor error: {exc}",
                flush=True,
            )

        time.sleep(
            interval
        )


def start_system_monitor(interval):

    if interval <= 0:
        return

    thread = threading.Thread(
        target=monitor_system,
        args=(interval,),
        daemon=True,
    )

    thread.start()


# =============================================================================
# Statistics
# =============================================================================

class DatasetStats:

    def __init__(self):

        self.rows = defaultdict(int)
        self.positive_rows = defaultdict(int)

        self.groups = defaultdict(int)
        self.positive_groups = defaultdict(int)


# =============================================================================
# Header validation
# =============================================================================

def validate_master_index_header(
    master_index,
):

    header = pd.read_csv(
        master_index,
        nrows=0,
    )

    required_columns = {
        "chunk_id",
        "row_idx_in_chunk",
        "y",
        "protein_group_id",
        "group_id",
    }

    missing = (
        required_columns
        - set(header.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing required columns in "
            f"{master_index}: "
            f"{sorted(missing)}"
        )


# =============================================================================
# PASS 1
# Discover group_id -> y
# =============================================================================

def discover_groups(
    master_index,
    read_rows,
):

    """
    Discover all unique group_id values and verify that each group_id has
    exactly one class label.

    Returns
    -------
    group_ids : np.ndarray[int64]
        Sorted unique group IDs.

    group_y : np.ndarray[uint8]
        Label associated with each group ID.
    """

    print()
    print(
        "PASS 1/2 - Discovering group_id values"
    )
    print("-" * 100)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We cannot simply keep a Python dictionary containing one entry per ROW.
    # We only keep one entry per UNIQUE group_id.
    #
    # This is much smaller than the augmented dataset because many BLOSUM rows
    # share the same group_id.
    # -------------------------------------------------------------------------

    group_to_y = {}

    total_rows = 0

    reader = pd.read_csv(
        master_index,
        usecols=[
            "group_id",
            "y",
        ],
        dtype={
            "group_id": "int64",
            "y": "uint8",
        },
        chunksize=read_rows,
    )

    for block_number, block in enumerate(
        reader,
        start=1,
    ):

        labels = set(
            block["y"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not labels.issubset(
            {0, 1}
        ):

            raise RuntimeError(
                f"Unexpected y values in "
                f"{master_index}: "
                f"{sorted(labels)}"
            )

        # ---------------------------------------------------------------------
        # Reduce current block to one record per (group_id, y).
        #
        # This is much cheaper than iterating over every augmented row.
        # ---------------------------------------------------------------------

        unique_pairs = (
            block[
                [
                    "group_id",
                    "y",
                ]
            ]
            .drop_duplicates()
        )

        # ---------------------------------------------------------------------
        # Detect group IDs having >1 label INSIDE this block.
        # ---------------------------------------------------------------------

        counts = (
            unique_pairs
            .groupby(
                "group_id",
                sort=False,
            )["y"]
            .nunique()
        )

        bad = counts[
            counts > 1
        ]

        if len(bad):

            examples = (
                bad.index
                .to_numpy()[:20]
                .tolist()
            )

            raise RuntimeError(
                "A group_id is associated with "
                "multiple y labels inside the same "
                f"read block. Examples: {examples}"
            )

        # ---------------------------------------------------------------------
        # Merge with groups discovered in previous blocks.
        # ---------------------------------------------------------------------

        unique_group = (
            unique_pairs[
                "group_id"
            ].to_numpy(
                dtype=np.int64,
                copy=False,
            )
        )

        unique_y = (
            unique_pairs[
                "y"
            ].to_numpy(
                dtype=np.uint8,
                copy=False,
            )
        )

        for gid, label in zip(
            unique_group,
            unique_y,
        ):

            gid = int(gid)
            label = int(label)

            previous = (
                group_to_y.get(gid)
            )

            if previous is None:

                group_to_y[
                    gid
                ] = label

            elif previous != label:

                raise RuntimeError(
                    f"group_id {gid} has "
                    f"inconsistent labels: "
                    f"{previous} and {label}"
                )

        total_rows += len(
            block
        )

        elapsed = (
            time.time()
            - start_time
        )

        if (
            block_number == 1
            or block_number % 20 == 0
        ):

            print(
                f"[PASS 1] "
                f"block={block_number:,} | "
                f"rows={total_rows:,} | "
                f"unique groups={len(group_to_y):,} | "
                f"speed="
                f"{total_rows / max(elapsed, 1e-9):,.0f} rows/s | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    if not group_to_y:

        raise RuntimeError(
            f"{master_index} contains no groups."
        )

    print()
    print(
        "Converting group table to arrays..."
    )

    group_ids = np.fromiter(
        group_to_y.keys(),
        dtype=np.int64,
        count=len(group_to_y),
    )

    group_y = np.fromiter(
        group_to_y.values(),
        dtype=np.uint8,
        count=len(group_to_y),
    )

    # -------------------------------------------------------------------------
    # Sort by group_id.
    #
    # This allows fast vectorized lookup with np.searchsorted during PASS 2.
    # -------------------------------------------------------------------------

    order = np.argsort(
        group_ids
    )

    group_ids = group_ids[
        order
    ]

    group_y = group_y[
        order
    ]

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"PASS 1 finished:"
    )
    print(
        f"  Rows scanned   : {total_rows:,}"
    )
    print(
        f"  Unique groups  : {len(group_ids):,}"
    )
    print(
        f"  Positive groups: "
        f"{int((group_y == 1).sum()):,}"
    )
    print(
        f"  Negative groups: "
        f"{int((group_y == 0).sum()):,}"
    )
    print(
        f"  Elapsed        : "
        f"{elapsed / 60:.2f} min"
    )

    return (
        group_ids,
        group_y,
    )


# =============================================================================
# Group-aware stratified assignment
# =============================================================================

def assign_group_class_split(
    group_indices,
    split_by_group,
    train_ratio,
    validation_ratio,
    rng,
):

    """
    Assign complete groups from one class to train/validation/test.

    The ratios are therefore defined over GROUPS, not individual rows.
    """

    indices = np.asarray(
        group_indices,
        dtype=np.int64,
    ).copy()

    rng.shuffle(
        indices
    )

    n = len(
        indices
    )

    n_train = int(
        round(
            n * train_ratio
        )
    )

    n_validation = int(
        round(
            n * validation_ratio
        )
    )

    n_train = min(
        n_train,
        n,
    )

    n_validation = min(
        n_validation,
        n - n_train,
    )

    train_end = n_train

    validation_end = (
        n_train
        + n_validation
    )

    split_by_group[
        indices[
            :train_end
        ]
    ] = TRAIN

    split_by_group[
        indices[
            train_end:
            validation_end
        ]
    ] = VALIDATION

    split_by_group[
        indices[
            validation_end:
        ]
    ] = TEST


def create_group_split(
    group_ids,
    group_y,
    train_ratio,
    validation_ratio,
    seed,
):

    """
    Create one split_id for every unique group_id.
    """

    print()
    print(
        "Creating stratified GROUP-AWARE split"
    )
    print("-" * 100)

    rng = np.random.default_rng(
        seed
    )

    split_by_group = np.full(
        len(group_ids),
        UNASSIGNED,
        dtype=np.uint8,
    )

    negative_groups = np.flatnonzero(
        group_y == 0
    )

    positive_groups = np.flatnonzero(
        group_y == 1
    )

    assign_group_class_split(
        negative_groups,
        split_by_group,
        train_ratio,
        validation_ratio,
        rng,
    )

    assign_group_class_split(
        positive_groups,
        split_by_group,
        train_ratio,
        validation_ratio,
        rng,
    )

    if np.any(
        split_by_group
        == UNASSIGNED
    ):

        raise RuntimeError(
            "Some group_id values were not "
            "assigned to a split."
        )

    # -------------------------------------------------------------------------
    # Group-level statistics
    # -------------------------------------------------------------------------

    print()

    for split in [
        TRAIN,
        VALIDATION,
        TEST,
    ]:

        mask = (
            split_by_group
            == split
        )

        n_groups = int(
            mask.sum()
        )

        positive = int(
            (
                mask
                & (group_y == 1)
            ).sum()
        )

        negative = (
            n_groups
            - positive
        )

        ratio = (
            n_groups
            / len(group_ids)
        )

        print(
            f"{split_name(split):10s}: "
            f"{n_groups:>12,} groups "
            f"({ratio:>6.2%}) | "
            f"positive={positive:,} | "
            f"negative={negative:,}"
        )

    return split_by_group


# =============================================================================
# Vectorized group lookup
# =============================================================================

def lookup_group_splits(
    query_group_ids,
    group_ids,
    split_by_group,
):

    """
    Vectorized:
        row group_id -> group table position -> split_id
    """

    query = np.asarray(
        query_group_ids,
        dtype=np.int64,
    )

    positions = np.searchsorted(
        group_ids,
        query,
    )

    valid = (
        positions
        < len(group_ids)
    )

    if not np.all(
        valid
    ):

        bad = query[
            ~valid
        ][:20]

        raise RuntimeError(
            "Some group_id values from PASS 2 "
            "were not discovered during PASS 1. "
            f"Examples: {bad.tolist()}"
        )

    resolved_ids = group_ids[
        positions
    ]

    exact = (
        resolved_ids
        == query
    )

    if not np.all(
        exact
    ):

        bad = query[
            ~exact
        ][:20]

        raise RuntimeError(
            "Some group_id values cannot be "
            "resolved exactly. "
            f"Examples: {bad.tolist()}"
        )

    return split_by_group[
        positions
    ]


# =============================================================================
# PASS 2
# Write master_split.csv
# =============================================================================

def write_master_split(
    master_index,
    group_ids,
    group_y,
    split_by_group,
    read_rows,
):

    output = (
        master_index.parent
        / "master_split.csv"
    )

    print()
    print(
        "PASS 2/2 - Writing group-aware master_split.csv"
    )
    print("-" * 100)

    if output.exists():

        print(
            f"Removing existing output: "
            f"{output}"
        )

        output.unlink()

    stats = DatasetStats()

    # -------------------------------------------------------------------------
    # Group-level statistics
    # -------------------------------------------------------------------------

    for split in [
        TRAIN,
        VALIDATION,
        TEST,
    ]:

        mask = (
            split_by_group
            == split
        )

        stats.groups[
            split
        ] = int(
            mask.sum()
        )

        stats.positive_groups[
            split
        ] = int(
            (
                mask
                & (group_y == 1)
            ).sum()
        )

    # -------------------------------------------------------------------------
    # Stream original master_index.
    # -------------------------------------------------------------------------

    reader = pd.read_csv(
        master_index,
        chunksize=read_rows,
        dtype={
            "chunk_id": "int32",
            "row_idx_in_chunk": "int32",
            "y": "uint8",
            "protein_group_id": "int64",
            "group_id": "int64",
        },
    )

    write_header = True

    total_rows = 0

    start_time = time.time()

    for block_number, block in enumerate(
        reader,
        start=1,
    ):

        if block.empty:
            continue

        labels = set(
            block[
                "y"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not labels.issubset(
            {0, 1}
        ):

            raise RuntimeError(
                f"Unexpected y values: "
                f"{sorted(labels)}"
            )

        # ---------------------------------------------------------------------
        # Resolve group_id -> split_id.
        # ---------------------------------------------------------------------

        split_ids = (
            lookup_group_splits(
                block[
                    "group_id"
                ].to_numpy(
                    dtype=np.int64,
                    copy=False,
                ),
                group_ids,
                split_by_group,
            )
        )

        block[
            "split_id"
        ] = split_ids

        # ---------------------------------------------------------------------
        # Row-level statistics.
        # ---------------------------------------------------------------------

        y = block[
            "y"
        ].to_numpy(
            dtype=np.uint8,
            copy=False,
        )

        for split in [
            TRAIN,
            VALIDATION,
            TEST,
        ]:

            mask = (
                split_ids
                == split
            )

            n_rows = int(
                mask.sum()
            )

            positive = int(
                y[
                    mask
                ].sum()
            )

            stats.rows[
                split
            ] += n_rows

            stats.positive_rows[
                split
            ] += positive

        # ---------------------------------------------------------------------
        # Write immediately.
        # ---------------------------------------------------------------------

        block.to_csv(
            output,
            mode=(
                "w"
                if write_header
                else "a"
            ),
            header=write_header,
            index=False,
        )

        write_header = False

        total_rows += len(
            block
        )

        elapsed = (
            time.time()
            - start_time
        )

        if (
            block_number == 1
            or block_number % 20 == 0
        ):

            print(
                f"[PASS 2] "
                f"block={block_number:,} | "
                f"rows={total_rows:,} | "
                f"speed="
                f"{total_rows / max(elapsed, 1e-9):,.0f} rows/s | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    if total_rows == 0:

        raise RuntimeError(
            f"{master_index} is empty."
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"PASS 2 finished: "
        f"{total_rows:,} rows | "
        f"{elapsed / 60:.2f} minutes"
    )

    return (
        output,
        stats,
        total_rows,
    )


# =============================================================================
# Output statistics
# =============================================================================

def validate_output(
    stats,
    total_rows,
    train_ratio,
    validation_ratio,
    test_ratio,
):

    print()
    print("=" * 100)
    print(
        "FINAL SPLIT STATISTICS"
    )
    print("=" * 100)

    expected = {
        TRAIN: train_ratio,
        VALIDATION: validation_ratio,
        TEST: test_ratio,
    }

    # -------------------------------------------------------------------------
    # GROUP LEVEL
    # -------------------------------------------------------------------------

    print()
    print(
        "GROUP-LEVEL distribution"
    )
    print("-" * 100)

    total_groups = sum(
        stats.groups.values()
    )

    for split in [
        TRAIN,
        VALIDATION,
        TEST,
    ]:

        groups = (
            stats.groups[
                split
            ]
        )

        positive = (
            stats.positive_groups[
                split
            ]
        )

        negative = (
            groups
            - positive
        )

        ratio = (
            groups
            / total_groups
            if total_groups
            else 0
        )

        prevalence = (
            positive
            / groups
            if groups
            else 0
        )

        print(
            f"{split_name(split):10s}: "
            f"{groups:>12,} "
            f"({ratio:>6.2%}) | "
            f"positive={positive:>10,} | "
            f"negative={negative:>12,} | "
            f"positive prevalence="
            f"{prevalence:.4%}"
        )

        # With enough groups this should be extremely close.
        if (
            total_groups >= 100
            and abs(
                ratio
                - expected[split]
            ) > 0.01
        ):

            print(
                f"WARNING: group-level "
                f"{split_name(split)} ratio "
                f"is {ratio:.2%}; expected "
                f"approximately "
                f"{expected[split]:.2%}."
            )

    # -------------------------------------------------------------------------
    # ROW LEVEL
    # -------------------------------------------------------------------------

    print()
    print(
        "ROW-LEVEL distribution"
    )
    print("-" * 100)

    counted_rows = sum(
        stats.rows.values()
    )

    if (
        counted_rows
        != total_rows
    ):

        raise RuntimeError(
            "Row statistics do not match "
            "written rows: "
            f"{counted_rows:,} != "
            f"{total_rows:,}"
        )

    for split in [
        TRAIN,
        VALIDATION,
        TEST,
    ]:

        rows = (
            stats.rows[
                split
            ]
        )

        positive = (
            stats.positive_rows[
                split
            ]
        )

        negative = (
            rows
            - positive
        )

        ratio = (
            rows
            / total_rows
            if total_rows
            else 0
        )

        prevalence = (
            positive
            / rows
            if rows
            else 0
        )

        print(
            f"{split_name(split):10s}: "
            f"{rows:>14,} "
            f"({ratio:>6.2%}) | "
            f"positive={positive:>12,} | "
            f"negative={negative:>14,} | "
            f"positive prevalence="
            f"{prevalence:.4%}"
        )

    print()
    print(
        "NOTE: exact 80/10/10 is enforced approximately "
        "at GROUP level, not at individual-row or physical-chunk level."
    )


# =============================================================================
# Process dataset
# =============================================================================

def process_dataset(
    master_index,
    train_ratio,
    validation_ratio,
    test_ratio,
    seed,
    read_rows,
):

    print_header(
        master_index.parent
    )

    validate_master_index_header(
        master_index
    )

    # =========================================================================
    # PASS 1
    # =========================================================================

    group_ids, group_y = (
        discover_groups(
            master_index,
            read_rows,
        )
    )

    # =========================================================================
    # Split unique groups
    # =========================================================================

    split_by_group = (
        create_group_split(
            group_ids,
            group_y,
            train_ratio,
            validation_ratio,
            seed,
        )
    )

    # =========================================================================
    # PASS 2
    # =========================================================================

    output, stats, total_rows = (
        write_master_split(
            master_index,
            group_ids,
            group_y,
            split_by_group,
            read_rows,
        )
    )

    # =========================================================================
    # Statistics
    # =========================================================================

    validate_output(
        stats,
        total_rows,
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    print()
    print(
        f"Saved: {output}"
    )
    print()


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate group-aware master_split.csv. "
            "All rows sharing a group_id are assigned "
            "to exactly one split."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing "
            "data_input_nn."
        ),
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.80,
        help="Training group ratio.",
    )

    parser.add_argument(
        "--validation",
        type=float,
        default=0.10,
        help="Validation group ratio.",
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.10,
        help="Test group ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--read-rows",
        type=int,
        default=500_000,
        help=(
            "CSV rows read at once. "
            "Default: 500000."
        ),
    )

    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=60,
        help=(
            "Seconds between resource reports. "
            "Use 0 to disable. "
            "Default: 60."
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # Validate arguments
    # =========================================================================

    if not np.isclose(
        args.train
        + args.validation
        + args.test,
        1.0,
    ):

        raise RuntimeError(
            "Split ratios must sum to 1."
        )

    if (
        args.train < 0
        or args.validation < 0
        or args.test < 0
    ):

        raise RuntimeError(
            "Split ratios cannot be negative."
        )

    if args.read_rows <= 0:

        raise RuntimeError(
            "--read-rows must be greater than 0."
        )

    # =========================================================================
    # Start monitoring
    # =========================================================================

    start_system_monitor(
        args.monitor_interval
    )

    # =========================================================================
    # Discover datasets
    # =========================================================================

    datasets = discover_datasets(
        args.root
    )

    if len(datasets) == 0:

        raise RuntimeError(
            "No datasets found."
        )

    print()
    print("=" * 100)
    print(
        f"Datasets found: "
        f"{len(datasets)}"
    )
    print("=" * 100)

    # =========================================================================
    # Process
    # =========================================================================

    for master_index in datasets:

        process_dataset(
            master_index,
            args.train,
            args.validation,
            args.test,
            args.seed,
            args.read_rows,
        )

    print()
    print("=" * 100)
    print(
        "Finished successfully."
    )
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()