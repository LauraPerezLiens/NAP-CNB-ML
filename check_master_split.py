#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MASTER SPLIT DEEP VALIDATION
===============================================================================

Deep integrity validation for the NAP-CNB master split.

Checks:

A) PHYSICAL PRIMARY DATA
   - master_index.csv exists
   - master_split.csv exists
   - physical chunk IDs are consecutive
   - chunk index files are valid
   - row_idx_in_chunk == 0..N-1
   - required NPY files exist
   - NPY row counts match physical index
   - y.npy matches index.csv
   - group_id.npy matches index.csv
   - protein_group_id.npy matches index.csv
   - y contains only 0/1

B) MASTER FILES
   - required columns exist
   - master_index and master_split have identical rows
   - split_id contains only 0/1/2
   - every physical chunk occurs exactly once
   - row_idx_in_chunk is valid
   - approximate 80/10/10 stratification per chunk/class

C) SPLIT POLICY
   - split is performed independently inside each physical chunk
   - y=0 rows are split approximately 80/10/10
   - y=1 rows are split approximately 80/10/10
   - group_id is NOT used to constrain the split
   - protein_group_id is NOT used to constrain the split
   - therefore the same group/protein may legitimately occur in multiple splits

D) SECONDARYBERT CROSS-MODAL INTEGRITY
   - corresponding *_SecondaryBERT_by_group directory exists
   - global group_id.npy exists
   - global protein_group_id.npy exists
   - SecondaryBERT IDs have equal lengths
   - SecondaryBERT group_id is strictly increasing
   - every master group_id exists in SecondaryBERT
   - master protein_group_id agrees with SecondaryBERT
   - SecondaryBERT lookup is unique by construction

All cross-modal checks are exhaustive. No sampling is used for IDs.

Large files are streamed and SecondaryBERT global arrays are memory-mapped.
===============================================================================
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import re
import time
import sys


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(
    "/data/nap/lperez_nn/data/data_input_nn"
)

TRAIN = 0
VALIDATION = 1
TEST = 2

EXPECTED_SPLITS = {
    TRAIN,
    VALIDATION,
    TEST,
}

SPLIT_NAMES = {
    TRAIN: "train",
    VALIDATION: "validation",
    TEST: "test",
}

INDEX_PATTERN = re.compile(
    r"chunk_(\d+)_index\.csv$"
)

READ_ROWS = 500_000

# =============================================================================
# Helpers
# =============================================================================

def fail(errors, message):

    errors.append(message)

    print(
        f"    ❌ {message}"
    )


def warn(warnings, message):

    warnings.append(message)

    print(
        f"    ⚠ {message}"
    )


def ok(message):

    print(
        f"    ✓ {message}"
    )


def load_npy(path):

    if not path.exists():
        return None

    return np.load(
        path,
        mmap_mode="r",
    )


def find_secondary_directory(
    primary_dataset_dir,
):
    """
    Find the SecondaryBERT_by_group directory corresponding to
    the current primary dataset.

    master_split.csv is expected inside a primary chunk directory,
    while SecondaryBERT is normally a sibling directory.
    """

    search_roots = [
        primary_dataset_dir,
        primary_dataset_dir.parent,
    ]

    candidates = set()

    for root in search_roots:

        if not root.is_dir():
            continue

        for path in root.glob(
            "*_SecondaryBERT_by_group"
        ):

            if path.is_dir():
                candidates.add(
                    path.resolve()
                )

    candidates = sorted(
        candidates
    )

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:

        raise RuntimeError(
            "Multiple SecondaryBERT directories found:\n"
            + "\n".join(
                f"        {path}"
                for path in candidates
            )
        )

    return candidates[0]


def validate_strictly_increasing_mmap(
    values,
    block_size=1_000_000,
):
    """
    Exhaustively verify that a 1-D mmap array is strictly increasing
    without loading the complete array into RAM.
    """

    if values.ndim != 1:
        return False

    n = len(values)

    if n <= 1:
        return True

    previous_last = None

    for start in range(
        0,
        n,
        block_size,
    ):

        end = min(
            start + block_size,
            n,
        )

        block = np.asarray(
            values[start:end]
        )

        if len(block) > 1:

            if np.any(
                block[1:] <= block[:-1]
            ):
                return False

        if (
            previous_last is not None
            and block[0] <= previous_last
        ):
            return False

        previous_last = block[-1]

    return True


def secondary_lookup(
    secondary_group_ids,
    query_group_ids,
):
    """
    Resolve group IDs against the sorted SecondaryBERT global group_id.npy.

    Returns
    -------
    positions : np.ndarray[int64]
        searchsorted positions.

    found : np.ndarray[bool]
        True where query_group_id exists exactly.
    """

    query_group_ids = np.asarray(
        query_group_ids,
        dtype=np.int64,
    )

    positions = np.searchsorted(
        secondary_group_ids,
        query_group_ids,
    )

    found = (
        positions
        < len(secondary_group_ids)
    )

    if found.any():

        valid_positions = positions[
            found
        ]

        found_indices = np.flatnonzero(
            found
        )

        exact = (
            np.asarray(
                secondary_group_ids[
                    valid_positions
                ]
            )
            == query_group_ids[
                found
            ]
        )

        found[
            found_indices
        ] = exact

    return positions, found


# =============================================================================
# Dataset validation
# =============================================================================

def validate_dataset(
    dataset_dir,
):

    start_time = time.time()

    master_index = (
        dataset_dir
        / "master_index.csv"
    )

    master_split = (
        dataset_dir
        / "master_split.csv"
    )

    errors = []
    warnings = []

    print()
    print(
        "=" * 100
    )

    print(
        f"DATASET: {dataset_dir}"
    )

    print(
        "=" * 100
    )

    # =========================================================================
    # Basic master files
    # =========================================================================

    if not master_index.exists():

        fail(
            errors,
            "master_index.csv missing",
        )

        return (
            False,
            errors,
            warnings,
        )

    if not master_split.exists():

        fail(
            errors,
            "master_split.csv missing",
        )

        return (
            False,
            errors,
            warnings,
        )

    ok(
        "master_index.csv exists"
    )

    ok(
        "master_split.csv exists"
    )

    # =========================================================================
    # SecondaryBERT store
    # =========================================================================

    print()
    print(
        "    Locating SecondaryBERT store..."
    )

    try:

        secondary_dir = (
            find_secondary_directory(
                dataset_dir
            )
        )

    except Exception as exc:

        fail(
            errors,
            (
                "Cannot identify SecondaryBERT "
                f"directory: {exc}"
            ),
        )

        return (
            False,
            errors,
            warnings,
        )

    if secondary_dir is None:

        fail(
            errors,
            (
                "No corresponding "
                "*_SecondaryBERT_by_group "
                "directory found"
            ),
        )

        return (
            False,
            errors,
            warnings,
        )

    ok(
        f"SecondaryBERT directory: "
        f"{secondary_dir.name}"
    )

    secondary_group_path = (
        secondary_dir
        / "group_id.npy"
    )

    secondary_protein_path = (
        secondary_dir
        / "protein_group_id.npy"
    )

    secondary_manifest = (
        secondary_dir
        / "chunks_manifest.csv"
    )

    secondary_chunks = (
        secondary_dir
        / "chunks"
    )

    for path, description in [
        (
            secondary_group_path,
            "SecondaryBERT group_id.npy",
        ),
        (
            secondary_protein_path,
            "SecondaryBERT protein_group_id.npy",
        ),
        (
            secondary_manifest,
            "SecondaryBERT chunks_manifest.csv",
        ),
    ]:

        if not path.exists():

            fail(
                errors,
                f"{description} missing: {path}",
            )

    if not secondary_chunks.is_dir():

        fail(
            errors,
            (
                "SecondaryBERT chunks directory "
                f"missing: {secondary_chunks}"
            ),
        )

    if errors:

        return (
            False,
            errors,
            warnings,
        )

    secondary_group_ids = np.load(
        secondary_group_path,
        mmap_mode="r",
    )

    secondary_protein_ids = np.load(
        secondary_protein_path,
        mmap_mode="r",
    )

    if secondary_group_ids.ndim != 1:

        fail(
            errors,
            (
                "SecondaryBERT group_id.npy "
                f"must be 1-D; found "
                f"{secondary_group_ids.shape}"
            ),
        )

    if secondary_protein_ids.ndim != 1:

        fail(
            errors,
            (
                "SecondaryBERT protein_group_id.npy "
                f"must be 1-D; found "
                f"{secondary_protein_ids.shape}"
            ),
        )

    if (
        len(secondary_group_ids)
        != len(secondary_protein_ids)
    ):

        fail(
            errors,
            (
                "SecondaryBERT global ID arrays "
                "have different lengths: "
                f"group={len(secondary_group_ids):,}, "
                f"protein={len(secondary_protein_ids):,}"
            ),
        )

    else:

        ok(
            (
                "SecondaryBERT global ID arrays "
                f"have matching length: "
                f"{len(secondary_group_ids):,}"
            )
        )

    if errors:

        return (
            False,
            errors,
            warnings,
        )

    print(
        "    Checking SecondaryBERT group_id "
        "strict ordering..."
    )

    if not validate_strictly_increasing_mmap(
        secondary_group_ids
    ):

        fail(
            errors,
            (
                "SecondaryBERT group_id.npy "
                "is not strictly increasing / unique"
            ),
        )

        return (
            False,
            errors,
            warnings,
        )

    ok(
        (
            "SecondaryBERT group_id is "
            "strictly increasing and unique"
        )
    )


    # =========================================================================
    # Physical primary chunks
    # =========================================================================

    physical_chunks = []

    for path in dataset_dir.glob(
        "chunk_*_index.csv"
    ):

        match = INDEX_PATTERN.match(
            path.name
        )

        if match:

            cid = int(
                match.group(1)
            )

            physical_chunks.append(
                (
                    cid,
                    path,
                )
            )

    physical_chunks.sort()

    if not physical_chunks:

        fail(
            errors,
            (
                "No physical "
                "chunk_*_index.csv files found"
            ),
        )

        return (
            False,
            errors,
            warnings,
        )

    physical_ids = [
        cid
        for cid, _ in physical_chunks
    ]

    ok(
        (
            f"Physical chunks found: "
            f"{len(physical_chunks):,} "
            f"({physical_ids[0]} -> "
            f"{physical_ids[-1]})"
        )
    )

    # =========================================================================
    # Consecutive physical chunk IDs
    # =========================================================================

    expected_ids = list(
        range(
            physical_ids[0],
            physical_ids[-1] + 1,
        )
    )

    if physical_ids != expected_ids:

        missing = sorted(
            set(expected_ids)
            - set(physical_ids)
        )

        fail(
            errors,
            (
                "Physical chunk IDs are not "
                f"consecutive. Missing: "
                f"{missing[:20]}"
            ),
        )

    else:

        ok(
            "Physical chunk IDs are consecutive"
        )

    # =========================================================================
    # Master headers
    # =========================================================================

    index_header = pd.read_csv(
        master_index,
        nrows=0,
    ).columns.tolist()

    split_header = pd.read_csv(
        master_split,
        nrows=0,
    ).columns.tolist()

    required_index = [
        "chunk_id",
        "row_idx_in_chunk",
        "y",
        "protein_group_id",
        "group_id",
    ]

    for col in required_index:

        if col not in index_header:

            fail(
                errors,
                (
                    "master_index.csv missing "
                    f"column: {col}"
                ),
            )

        if col not in split_header:

            fail(
                errors,
                (
                    "master_split.csv missing "
                    f"column: {col}"
                ),
            )

    if "split_id" not in split_header:

        fail(
            errors,
            (
                "master_split.csv "
                "missing split_id"
            ),
        )

    if errors:

        return (
            False,
            errors,
            warnings,
        )

    ok(
        "Required columns exist"
    )

    # =========================================================================
    # Global counters
    # =========================================================================

    total_rows = 0

    global_split = defaultdict(
        int
    )

    global_positive_split = defaultdict(
        int
    )

    global_negative_split = defaultdict(
        int
    )

    total_positive = 0
    total_negative = 0

    validated_ids = set()

    secondary_rows_checked = 0
    secondary_missing_rows = 0
    secondary_protein_mismatch_rows = 0

    # =========================================================================
    # Physical chunk validation
    # =========================================================================

    print()
    print(
        "    Validating physical primary chunks..."
    )

    for number, (
        cid,
        index_file,
    ) in enumerate(
        physical_chunks,
        start=1,
    ):

        prefix = (
            f"chunk_{cid:06d}"
        )

        physical_index = pd.read_csv(
            index_file,
            dtype={
                "chunk_id": "int32",
                "row_idx_in_chunk": "int32",
                "y": "uint8",
                "protein_group_id": "int64",
                "group_id": "int64",
            },
        )

        n = len(
            physical_index
        )

        if n == 0:

            fail(
                errors,
                (
                    f"Chunk {cid}: "
                    "empty physical index"
                ),
            )

            continue

        # ---------------------------------------------------------------------
        # Internal chunk ID
        # ---------------------------------------------------------------------

        ids = physical_index[
            "chunk_id"
        ].unique()

        if (
            len(ids) != 1
            or int(ids[0]) != cid
        ):

            fail(
                errors,
                (
                    f"Chunk {cid}: invalid "
                    "chunk_id inside physical "
                    f"index: {ids.tolist()}"
                ),
            )

        # ---------------------------------------------------------------------
        # row_idx_in_chunk
        # ---------------------------------------------------------------------

        row_idx = physical_index[
            "row_idx_in_chunk"
        ].to_numpy()

        expected_row_idx = np.arange(
            n,
            dtype=row_idx.dtype,
        )

        if not np.array_equal(
            row_idx,
            expected_row_idx,
        ):

            fail(
                errors,
                (
                    f"Chunk {cid}: invalid "
                    "row_idx_in_chunk"
                ),
            )

        # ---------------------------------------------------------------------
        # Physical NPY files
        # ---------------------------------------------------------------------

        npy_files = {
            "X":
                dataset_dir
                / f"{prefix}_X.npy",

            "y":
                dataset_dir
                / f"{prefix}_y.npy",

            "pos":
                dataset_dir
                / f"{prefix}_pos.npy",

            "group_id":
                dataset_dir
                / f"{prefix}_group_id.npy",

            "protein_group_id":
                dataset_dir
                / (
                    f"{prefix}_"
                    "protein_group_id.npy"
                ),
        }

        arrays = {}

        for name, path in (
            npy_files.items()
        ):

            if not path.exists():

                fail(
                    errors,
                    (
                        f"Chunk {cid}: "
                        f"missing {path.name}"
                    ),
                )

                arrays[name] = None

                continue

            try:

                arrays[name] = (
                    load_npy(path)
                )

            except Exception as exc:

                fail(
                    errors,
                    (
                        f"Chunk {cid}: cannot "
                        f"load {path.name}: "
                        f"{exc}"
                    ),
                )

                arrays[name] = None

                continue

            if (
                arrays[name].shape[0]
                != n
            ):

                fail(
                    errors,
                    (
                        f"Chunk {cid}: "
                        f"{path.name} has "
                        f"{arrays[name].shape[0]:,} "
                        f"rows but index has "
                        f"{n:,}"
                    ),
                )

        # ---------------------------------------------------------------------
        # Compare physical arrays with index
        # ---------------------------------------------------------------------

        if arrays.get(
            "y"
        ) is not None:

            if not np.array_equal(
                physical_index[
                    "y"
                ].to_numpy(),
                np.asarray(
                    arrays["y"]
                ),
            ):

                fail(
                    errors,
                    (
                        f"Chunk {cid}: y.npy "
                        "does not match index.csv"
                    ),
                )

        if arrays.get(
            "group_id"
        ) is not None:

            if not np.array_equal(
                physical_index[
                    "group_id"
                ].to_numpy(),
                np.asarray(
                    arrays["group_id"]
                ),
            ):

                fail(
                    errors,
                    (
                        f"Chunk {cid}: "
                        "group_id.npy does not "
                        "match index.csv"
                    ),
                )

        if arrays.get(
            "protein_group_id"
        ) is not None:

            if not np.array_equal(
                physical_index[
                    "protein_group_id"
                ].to_numpy(),
                np.asarray(
                    arrays[
                        "protein_group_id"
                    ]
                ),
            ):

                fail(
                    errors,
                    (
                        f"Chunk {cid}: "
                        "protein_group_id.npy "
                        "does not match index.csv"
                    ),
                )

        # ---------------------------------------------------------------------
        # y validity
        # ---------------------------------------------------------------------

        y = physical_index[
            "y"
        ].to_numpy()

        unique_y = set(
            np.unique(y).tolist()
        )

        if not unique_y.issubset(
            {0, 1}
        ):

            fail(
                errors,
                (
                    f"Chunk {cid}: invalid "
                    f"y values: "
                    f"{sorted(unique_y)}"
                ),
            )

        validated_ids.add(
            cid
        )

        if (
            number == 1
            or number % 100 == 0
            or number
            == len(physical_chunks)
        ):

            print(
                (
                    "    Physical validation "
                    f"{number:,}/"
                    f"{len(physical_chunks):,} "
                    f"| chunk={cid} "
                    f"| rows={n:,}"
                )
            )

    # =========================================================================
    # Stream master_index + master_split
    # =========================================================================

    print()
    print(
        "    Comparing master_index.csv and "
        "master_split.csv row by row..."
    )

    print(
        "    Checking SecondaryBERT coverage "
        "and Primary/Secondary ID consistency..."
    )

    usecols_index = [
        "chunk_id",
        "row_idx_in_chunk",
        "y",
        "protein_group_id",
        "group_id",
    ]

    usecols_split = (
        usecols_index
        + ["split_id"]
    )

    dtype_index = {
        "chunk_id": "int32",
        "row_idx_in_chunk": "int32",
        "y": "uint8",
        "protein_group_id": "int64",
        "group_id": "int64",
    }

    dtype_split = {
        **dtype_index,
        "split_id": "uint8",
    }

    index_reader = pd.read_csv(
        master_index,
        usecols=usecols_index,
        dtype=dtype_index,
        chunksize=READ_ROWS,
    )

    split_reader = pd.read_csv(
        master_split,
        usecols=usecols_split,
        dtype=dtype_split,
        chunksize=READ_ROWS,
    )

    block_number = 0

    while True:

        try:

            index_block = next(
                index_reader
            )

            index_finished = False

        except StopIteration:

            index_block = None
            index_finished = True

        try:

            split_block = next(
                split_reader
            )

            split_finished = False

        except StopIteration:

            split_block = None
            split_finished = True

        if (
            index_finished
            and split_finished
        ):
            break

        if (
            index_finished
            != split_finished
        ):

            fail(
                errors,
                (
                    "master_index.csv and "
                    "master_split.csv have "
                    "different number of rows"
                ),
            )

            break

        block_number += 1

        if (
            len(index_block)
            != len(split_block)
        ):

            fail(
                errors,
                (
                    f"Block {block_number}: "
                    "different block sizes: "
                    f"index={len(index_block):,}, "
                    f"split={len(split_block):,}"
                ),
            )

            break

        # ---------------------------------------------------------------------
        # Exact master_index vs master_split
        # ---------------------------------------------------------------------

        for col in usecols_index:

            a = index_block[
                col
            ].to_numpy()

            b = split_block[
                col
            ].to_numpy()

            if not np.array_equal(
                a,
                b,
            ):

                mismatch = np.flatnonzero(
                    a != b
                )

                first = int(
                    mismatch[0]
                )

                global_row = (
                    total_rows
                    + first
                )

                fail(
                    errors,
                    (
                        f"Block {block_number}: "
                        f"column '{col}' differs "
                        f"at global row "
                        f"{global_row:,}. "
                        f"index={a[first]}, "
                        f"split={b[first]}"
                    ),
                )

                break

        # ---------------------------------------------------------------------
        # split_id validity
        # ---------------------------------------------------------------------

        split_ids = split_block[
            "split_id"
        ].to_numpy(
            dtype=np.uint8,
            copy=False,
        )

        unique_split = set(
            np.unique(
                split_ids
            ).tolist()
        )

        invalid = (
            unique_split
            - EXPECTED_SPLITS
        )

        if invalid:

            fail(
                errors,
                (
                    f"Block {block_number}: "
                    "invalid split_id values: "
                    f"{sorted(invalid)}"
                ),
            )

        # ---------------------------------------------------------------------
        # SecondaryBERT exact lookup
        # ---------------------------------------------------------------------

        master_group = split_block[
            "group_id"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        master_protein = split_block[
            "protein_group_id"
        ].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        positions, found = (
            secondary_lookup(
                secondary_group_ids,
                master_group,
            )
        )

        secondary_rows_checked += len(
            master_group
        )

        missing_count = int(
            (~found).sum()
        )

        secondary_missing_rows += (
            missing_count
        )

        if missing_count:

            missing_indices = (
                np.flatnonzero(
                    ~found
                )
            )

            examples = master_group[
                missing_indices[:20]
            ].tolist()

            fail(
                errors,
                (
                    f"Block {block_number}: "
                    f"{missing_count:,} rows "
                    "have group_id absent from "
                    "SecondaryBERT. "
                    f"Examples: {examples}"
                ),
            )

        # ---------------------------------------------------------------------
        # Secondary protein_group_id agreement
        # ---------------------------------------------------------------------

        if found.any():

            found_indices = (
                np.flatnonzero(
                    found
                )
            )

            found_positions = positions[
                found
            ]

            secondary_pg = np.asarray(
                secondary_protein_ids[
                    found_positions
                ]
            )

            master_pg_found = (
                master_protein[
                    found
                ]
            )

            protein_match = (
                secondary_pg
                == master_pg_found
            )

            mismatch_count = int(
                (~protein_match).sum()
            )

            secondary_protein_mismatch_rows += (
                mismatch_count
            )

            if mismatch_count:

                bad_local = (
                    np.flatnonzero(
                        ~protein_match
                    )
                )

                examples = []

                for local_index in (
                    bad_local[:10]
                ):

                    original_index = (
                        found_indices[
                            local_index
                        ]
                    )

                    examples.append(
                        (
                            int(
                                master_group[
                                    original_index
                                ]
                            ),
                            int(
                                master_protein[
                                    original_index
                                ]
                            ),
                            int(
                                secondary_pg[
                                    local_index
                                ]
                            ),
                        )
                    )

                fail(
                    errors,
                    (
                        f"Block {block_number}: "
                        f"{mismatch_count:,} "
                        "protein_group_id "
                        "mismatches between "
                        "Primary and Secondary. "
                        "(group_id, primary_pg, "
                        "secondary_pg) examples: "
                        f"{examples}"
                    ),
                )

        # ---------------------------------------------------------------------
        # Global statistics
        # ---------------------------------------------------------------------

        y = split_block[
            "y"
        ].to_numpy()

        for sid in [
            TRAIN,
            VALIDATION,
            TEST,
        ]:

            mask = (
                split_ids == sid
            )

            n_split = int(
                mask.sum()
            )

            positives = int(
                y[mask].sum()
            )

            negatives = (
                n_split
                - positives
            )

            global_split[
                sid
            ] += n_split

            global_positive_split[
                sid
            ] += positives

            global_negative_split[
                sid
            ] += negatives

        total_positive += int(
            y.sum()
        )

        total_negative += (
            len(y)
            - int(y.sum())
        )

        total_rows += len(
            split_block
        )

        if (
            block_number == 1
            or block_number % 50 == 0
        ):

            print(
                (
                    "    Master validation "
                    f"block={block_number:,} "
                    f"| rows={total_rows:,}"
                )
            )

    # =========================================================================
    # Cross-modal summary
    # =========================================================================

    print()
    print(
        "    SECONDARYBERT CROSS-MODAL CHECK"
    )

    print(
        "    " + "-" * 70
    )

    print(
        (
            "      master rows checked       : "
            f"{secondary_rows_checked:,}"
        )
    )

    print(
        (
            "      missing Secondary group   : "
            f"{secondary_missing_rows:,}"
        )
    )

    print(
        (
            "      protein ID mismatches     : "
            f"{secondary_protein_mismatch_rows:,}"
        )
    )

    if secondary_missing_rows == 0:

        ok(
            (
                "Every master group_id resolves "
                "to SecondaryBERT"
            )
        )

    if (
        secondary_protein_mismatch_rows
        == 0
    ):

        ok(
            (
                "Primary and Secondary "
                "protein_group_id agree exactly"
            )
        )

    ok(
        (
            "group_id/protein_group_id overlap between "
            "splits is allowed by the selected split policy"
        )
    )

    # =========================================================================
    # Validate stratification per physical chunk/class
    # =========================================================================

    print()
    print(
        (
            "    Validating stratification "
            "per physical chunk..."
        )
    )

    reader = pd.read_csv(
        master_split,
        usecols=[
            "chunk_id",
            "row_idx_in_chunk",
            "y",
            "split_id",
        ],
        dtype={
            "chunk_id": "int32",
            "row_idx_in_chunk": "int32",
            "y": "uint8",
            "split_id": "uint8",
        },
        chunksize=READ_ROWS,
    )

    pending = None

    seen_chunks = set()

    processed_chunks = 0

    def validate_split_chunk(
        chunk,
    ):

        nonlocal processed_chunks

        cid = int(
            chunk[
                "chunk_id"
            ].iloc[0]
        )

        if cid in seen_chunks:

            fail(
                errors,
                (
                    f"Chunk {cid} appears in "
                    "multiple separate blocks "
                    "in master_split.csv"
                ),
            )

            return

        seen_chunks.add(
            cid
        )

        row_idx = chunk[
            "row_idx_in_chunk"
        ].to_numpy()

        if not np.array_equal(
            row_idx,
            np.arange(
                len(chunk),
                dtype=row_idx.dtype,
            ),
        ):

            fail(
                errors,
                (
                    f"Chunk {cid}: master_split "
                    "row_idx_in_chunk is not "
                    "0..n-1"
                ),
            )

        for cls in [
            0,
            1,
        ]:

            class_rows = chunk[
                chunk["y"] == cls
            ]

            n_class = len(
                class_rows
            )

            if n_class == 0:
                continue

            counts = (
                class_rows[
                    "split_id"
                ]
                .value_counts()
                .to_dict()
            )

            n_train = int(
                counts.get(
                    TRAIN,
                    0,
                )
            )

            n_val = int(
                counts.get(
                    VALIDATION,
                    0,
                )
            )

            n_test = int(
                counts.get(
                    TEST,
                    0,
                )
            )

            expected_train = (
                0.80 * n_class
            )

            expected_val = (
                0.10 * n_class
            )

            expected_test = (
                0.10 * n_class
            )

            # The selected split policy is row-level stratification
            # independently for y=0 and y=1 inside every physical chunk.
            # Integer rounding can produce deviations of approximately one row.

            if abs(
                n_train
                - expected_train
            ) > 1.1:

                fail(
                    errors,
                    (
                        f"Chunk {cid}, y={cls}: "
                        "train count suspicious: "
                        f"{n_train}/{n_class}"
                    ),
                )

            if abs(
                n_val
                - expected_val
            ) > 1.1:

                fail(
                    errors,
                    (
                        f"Chunk {cid}, y={cls}: "
                        "validation count "
                        f"suspicious: "
                        f"{n_val}/{n_class}"
                    ),
                )

            if abs(
                n_test
                - expected_test
            ) > 1.1:

                fail(
                    errors,
                    (
                        f"Chunk {cid}, y={cls}: "
                        "test count suspicious: "
                        f"{n_test}/{n_class}"
                    ),
                )

        processed_chunks += 1

    for block in reader:

        if pending is not None:

            block = pd.concat(
                [
                    pending,
                    block,
                ],
                ignore_index=True,
            )

            pending = None

        last_cid = int(
            block[
                "chunk_id"
            ].iloc[-1]
        )

        tail_mask = (
            block["chunk_id"]
            == last_cid
        )

        complete = block[
            ~tail_mask
        ]

        pending = block[
            tail_mask
        ].copy()

        for _, chunk in (
            complete.groupby(
                "chunk_id",
                sort=False,
            )
        ):

            validate_split_chunk(
                chunk
            )

    if (
        pending is not None
        and len(pending)
    ):

        validate_split_chunk(
            pending
        )

    if (
        seen_chunks
        != set(physical_ids)
    ):

        missing = (
            set(physical_ids)
            - seen_chunks
        )

        extra = (
            seen_chunks
            - set(physical_ids)
        )

        fail(
            errors,
            (
                "Chunk set mismatch in "
                "master_split. "
                f"Missing={sorted(missing)[:20]}, "
                f"extra={sorted(extra)[:20]}"
            ),
        )

    else:

        ok(
            (
                f"All {len(seen_chunks):,} "
                "physical chunks present "
                "exactly once in master_split"
            )
        )

    # =========================================================================
    # Global statistics
    # =========================================================================

    print()
    print(
        "-" * 100
    )

    print(
        "GLOBAL STATISTICS"
    )

    print(
        "-" * 100
    )

    print(
        f"Rows      : {total_rows:,}"
    )

    if total_rows:

        print(
            (
                f"Positive  : "
                f"{total_positive:,} "
                f"("
                f"{100 * total_positive / total_rows:.4f}"
                f"%)"
            )
        )

        print(
            (
                f"Negative  : "
                f"{total_negative:,} "
                f"("
                f"{100 * total_negative / total_rows:.4f}"
                f"%)"
            )
        )

    print()

    for sid, name in [
        (
            TRAIN,
            "Train",
        ),
        (
            VALIDATION,
            "Validation",
        ),
        (
            TEST,
            "Test",
        ),
    ]:

        n = global_split[
            sid
        ]

        pos = (
            global_positive_split[
                sid
            ]
        )

        neg = (
            global_negative_split[
                sid
            ]
        )

        if n == 0:

            print(
                f"{name:10s}: 0"
            )

            continue

        print(
            (
                f"{name:10s}: "
                f"{n:>12,} "
                f"("
                f"{100*n/total_rows:6.2f}%"
                f") | "
                f"positive={pos:>10,} "
                f"("
                f"{100*pos/n:6.2f}%"
                f") | "
                f"negative={neg:>12,}"
            )
        )

    print()

    print(
        "CROSS-MODAL INTEGRITY"
    )

    print(
        (
            f"Master rows checked       : "
            f"{secondary_rows_checked:,}"
        )
    )

    print(
        (
            f"Missing Secondary rows    : "
            f"{secondary_missing_rows:,}"
        )
    )

    print(
        (
            f"Protein ID mismatches     : "
            f"{secondary_protein_mismatch_rows:,}"
        )
    )

    print(
        "Split overlap policy       : group/protein overlap allowed"
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} min"
    )

    print()

    # =========================================================================
    # Result
    # =========================================================================

    if errors:

        print(
            "❌ DATASET FAILED"
        )

        print(
            f"   Errors: "
            f"{len(errors):,}"
        )

        for error in (
            errors[:30]
        ):

            print(
                f"   - {error}"
            )

        if len(errors) > 30:

            print(
                (
                    "   ... and "
                    f"{len(errors)-30:,} more"
                )
            )

        return (
            False,
            errors,
            warnings,
        )

    print(
        "✅ DATASET PASSED ALL CHECKS"
    )

    return (
        True,
        errors,
        warnings,
    )


# =============================================================================
# Discover datasets
# =============================================================================

datasets = sorted(
    path.parent
    for path in ROOT.rglob(
        "master_split.csv"
    )
)

print(
    "=" * 100
)

print(
    "MASTER SPLIT DEEP VALIDATION"
)

print(
    "=" * 100
)

print(
    f"Root: {ROOT}"
)

print(
    f"Datasets found: {len(datasets)}"
)

if not datasets:

    print()
    print(
        "❌ No master_split.csv files found."
    )

    sys.exit(1)


# =============================================================================
# Validate all datasets
# =============================================================================

results = []

for dataset_dir in datasets:

    passed, errors, warnings = (
        validate_dataset(
            dataset_dir
        )
    )

    results.append(
        (
            dataset_dir,
            passed,
            len(errors),
            len(warnings),
        )
    )


# =============================================================================
# Final summary
# =============================================================================

print()
print()

print(
    "=" * 100
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 100
)

for (
    path,
    passed,
    n_errors,
    n_warnings,
) in results:

    try:

        relative = (
            path.relative_to(
                ROOT
            )
        )

    except ValueError:

        relative = path

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        (
            f"{status:4s} | "
            f"errors={n_errors:<4} | "
            f"warnings={n_warnings:<4} | "
            f"{relative}"
        )
    )


failed = [
    result
    for result in results
    if not result[1]
]

print()

print(
    (
        f"Passed: "
        f"{len(results)-len(failed)}"
        f"/{len(results)}"
    )
)

print(
    (
        f"Failed: "
        f"{len(failed)}"
        f"/{len(results)}"
    )
)

if failed:

    print()
    print(
        "❌ VALIDATION FAILED."
    )

    sys.exit(1)

print()
print(
    "✅ ALL MASTER SPLITS PASSED."
)

sys.exit(0)