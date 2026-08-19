#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding QC

Quality control for ProtT5 and SecondaryBERT datasets.

Checks

- folder structure
- required files
- chunk numbering
- array shapes
- dtype
- NaN / Inf
- labels
- positions
- CSV integrity
- ID coverage

Author:
Laura Pérez Liens
"""

from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import argparse
import csv
import re

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PRIMARY_DIM = 1024
SECONDARY_SHAPE = (25, 768)

EXPECTED_DTYPES = {

    "X": np.float32,

    "X_ss": np.float32,

    "y": np.int8,

    "pos": np.float32,

    "group_id": np.int64,

    "protein_group_id": np.int64

}

PRIMARY_FILES = {

    "X": "_X.npy",

    "y": "_y.npy",

    "pos": "_pos.npy",

    "group_id": "_group_id.npy",

    "protein_group_id": "_protein_group_id.npy",

    "metadata": "_metadata.csv",

    "index": "_index.csv"

}

SECONDARY_FILES = {

    "X_ss": "_X_ss.npy",

    "group_id": "_group_id.npy",

    "protein_group_id": "_protein_group_id.npy"

}

ALLOWED_LABELS = {0, 1}
# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Issue:

    dataset: str
    section: str
    level: str
    message: str
    path: str = ""


@dataclass
class DatasetSummary:

    dataset: str

    primary_chunks: int = 0
    secondary_chunks: int = 0

    primary_rows: int = 0
    secondary_rows: int = 0

    positives: int = 0
    negatives: int = 0

    result: str = "PASS"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

CHUNK_RE = re.compile(r"chunk_(\d{6})")


def np_load(path):

    return np.load(path, mmap_mode="r")


def chunk_number(name):

    m = CHUNK_RE.search(name)

    if m:

        return int(m.group(1))

    return None


def add_issue(issues,
              dataset,
              section,
              level,
              message,
              path=""):

    issues.append(

        Issue(

            dataset,

            section,

            level,

            message,

            str(path)

        )

    )


def find_chunk_files(directory,
                     suffix):
    """
    Return

        prefix -> file

    matching one suffix.

    Prevent

        group_id

    matching

        protein_group_id
    """

    files = {}

    for file in sorted(Path(directory).glob("chunk_*")):

        name = file.name

        if not name.endswith(suffix):
            continue

        if (
            suffix == "_group_id.npy"
            and
            name.endswith("_protein_group_id.npy")
        ):
            continue

        prefix = name[:-len(suffix)]

        files[prefix] = file

    return files


def discover_datasets(root):
    """
    Automatically discover datasets.
    """

    datasets = []

    root = Path(root)

    for directory in root.rglob("*"):

        if not directory.is_dir():
            continue

        primary = list(

            directory.glob("*ProtT5_chunks")

        )

        secondary = list(

            directory.glob("*SecondaryBERT_by_group")

        )

        if len(primary) != 1:
            continue

        if len(secondary) != 1:
            continue

        datasets.append(

            {

                "name":

                    str(

                        directory.relative_to(root)

                    ),

                "primary":

                    primary[0],

                "secondary":

                    secondary[0]

            }

        )

    return sorted(

        datasets,

        key=lambda x: x["name"]

    )

# =============================================================================
# GENERIC QC FUNCTIONS
# =============================================================================

def check_chunk_sequence(prefixes):
    """
    Check that chunk numbering is consecutive.
    """

    numbers = sorted(
        chunk_number(p)
        for p in prefixes
    )

    if not numbers:
        return [], []

    expected = set(
        range(numbers[0], numbers[-1] + 1)
    )

    missing = sorted(
        expected - set(numbers)
    )

    duplicated = [

        n

        for n, c in Counter(numbers).items()

        if c > 1

    ]

    return missing, duplicated


def check_required_files(dataset,
                         directory,
                         required,
                         issues):
    """
    Check that every chunk contains every required file.
    """

    maps = {}

    for key, suffix in required.items():

        maps[key] = find_chunk_files(
            directory,
            suffix
        )

    reference = sorted(
        maps[list(required.keys())[0]]
    )

    for key in required:

        current = set(maps[key])

        missing = set(reference) - current
        extra = current - set(reference)

        if missing:

            add_issue(
                issues,
                dataset,
                "files",
                "FAIL",
                f"Missing {key} files"
            )

        if extra:

            add_issue(
                issues,
                dataset,
                "files",
                "WARNING",
                f"Extra {key} files"
            )

    return maps, reference


def check_array(name,
                path,
                expected_shape,
                issues,
                dataset):
    """
    Generic numpy validation.
    """

    arr = np_load(path)

    if arr.shape[0] == 0:

        add_issue(
            issues,
            dataset,
            name,
            "FAIL",
            "Empty array",
            path
        )

        return arr

    expected_dtype = EXPECTED_DTYPES.get(name)

    if expected_dtype is not None:

        if np.dtype(arr.dtype) != np.dtype(expected_dtype):

            add_issue(
                issues,
                dataset,
                name,
                "WARNING",
                f"Unexpected dtype ({arr.dtype})",
                path
            )

    if len(arr.shape) != len(expected_shape):

        add_issue(
            issues,
            dataset,
            name,
            "FAIL",
            f"Wrong shape {arr.shape}",
            path
        )

    else:

        for dim, expected in enumerate(expected_shape):

            if expected is None:
                continue

            if arr.shape[dim] != expected:

                add_issue(
                    issues,
                    dataset,
                    name,
                    "FAIL",
                    f"Wrong shape {arr.shape}",
                    path
                )

                break

  
        # ----------------------------------------------------------
    # Numerical validation
    # ----------------------------------------------------------

    if np.issubdtype(arr.dtype, np.number):

        if np.isnan(arr).any():

            add_issue(
                issues,
                dataset,
                name,
                "FAIL",
                "NaN detected",
                path
            )

        if np.isinf(arr).any():

            add_issue(
                issues,
                dataset,
                name,
                "FAIL",
                "Inf detected",
                path
            )

    return arr


def check_csv(path,
              expected_rows,
              dataset,
              section,
              issues):
    """
    Validate CSV file without loading it entirely into memory.
    """

    rows = 0
    found_nan = False
    found_duplicates = False

    for chunk in pd.read_csv(path, chunksize=100000):

        rows += len(chunk)

        if not found_nan and chunk.isna().any().any():

            found_nan = True

            add_issue(
                issues,
                dataset,
                section,
                "WARNING",
                "NaN values",
                path
            )

        if not found_duplicates and chunk.duplicated().any():

            found_duplicates = True

            add_issue(
                issues,
                dataset,
                section,
                "WARNING",
                "Duplicated rows",
                path
            )

    if rows != expected_rows:

        add_issue(
            issues,
            dataset,
            section,
            "FAIL",
            f"Wrong number of rows ({rows}, expected {expected_rows})",
            path
        )


def check_labels(labels,
                 dataset,
                 issues):

    values, counts = np.unique(
        labels,
        return_counts=True
    )

    invalid = set(values) - set(ALLOWED_LABELS)

    if invalid:

        add_issue(
            issues,
            dataset,
            "labels",
            "FAIL",
            f"Invalid labels {sorted(invalid)}"
        )

    distribution = {

        int(v): int(c)

        for v, c in zip(values, counts)

    }

    return distribution


def check_positions(pos,
                    dataset,
                    issues):

    if np.isnan(pos).any():

        add_issue(
            issues,
            dataset,
            "positions",
            "FAIL",
            "NaN values"
        )

    if np.isinf(pos).any():

        add_issue(
            issues,
            dataset,
            "positions",
            "FAIL",
            "Inf values"
        )

    finite = pos[np.isfinite(pos)]

    if finite.size:

        if finite.min() < -1 or finite.max() > 1:

            add_issue(
                issues,
                dataset,
                "positions",
                "WARNING",
                f"Range [{finite.min()}, {finite.max()}]"
            )


def concatenate(files):
    """
    Concatenate several numpy files.
    """

    arrays = []

    for file in files:

        arrays.append(

            np.asarray(

                np_load(file)

            ).reshape(-1)

        )

    if not arrays:

        return np.array([])

    return np.concatenate(arrays)


def compare_ids(primary,
                secondary):

    p = np.unique(primary.astype(np.int64))
    s = np.unique(secondary.astype(np.int64))

    return {

    "primary_unique": len(p),

    "secondary_unique": len(s),

    "missing": np.setdiff1d(
        p,
        s
    ),

    "extra": np.setdiff1d(
        s,
        p
    )

}

# =============================================================================
# PRIMARY DATASET
# =============================================================================

def check_primary_dataset(dataset_name,
                          primary_dir,
                          issues):

    summary = {
        "chunks": 0,
        "rows": 0,
        "labels": Counter(),
        "group_id_files": [],
        "protein_group_id_files": [],
        "chunk_sizes": []
}

    file_maps, prefixes = check_required_files(

        dataset_name,

        primary_dir,

        PRIMARY_FILES,

        issues

    )

    summary["chunks"] = len(prefixes)

    missing, duplicated = check_chunk_sequence(prefixes)

    if missing:

        add_issue(
            issues,
            dataset_name,
            "chunks",
            "FAIL",
            f"Missing chunks {missing}"
        )

    if duplicated:

        add_issue(
            issues,
            dataset_name,
            "chunks",
            "FAIL",
            f"Duplicated chunks {duplicated}"
        )

    for prefix in prefixes:

        # ----------------------------------------------------------
        # Embeddings
        # ----------------------------------------------------------

        x = check_array(

            "X",

            file_maps["X"][prefix],

            (None, PRIMARY_DIM),

            issues,

            dataset_name

        )

        n = x.shape[0]

        summary["rows"] += n

        summary["chunk_sizes"].append(n)

        # ----------------------------------------------------------
        # Labels
        # ----------------------------------------------------------

        y = np_load(

            file_maps["y"][prefix]

        )

        if len(y) != n:

            add_issue(
                issues,
                dataset_name,
                "labels",
                "FAIL",
                "Wrong label length"
            )

        labels = check_labels(

            np.asarray(y),

            dataset_name,

            issues

        )

        summary["labels"].update(labels)

        # ----------------------------------------------------------
        # Positions
        # ----------------------------------------------------------

        pos = np_load(

            file_maps["pos"][prefix]

        )

        if len(pos) != n:

            add_issue(
                issues,
                dataset_name,
                "positions",
                "FAIL",
                "Wrong position length"
            )

        check_positions(

            np.asarray(pos),

            dataset_name,

            issues

        )

        # ----------------------------------------------------------
        # group_id
        # ----------------------------------------------------------

        group = check_array(

            "group_id",

            file_maps["group_id"][prefix],

            (None,),

            issues,

            dataset_name

        )

        if len(group) != n:

            add_issue(
                issues,
                dataset_name,
                "group_id",
                "FAIL",
                "Wrong group_id length"
            )

        summary["group_id_files"].append(

            file_maps["group_id"][prefix]

        )

        # ----------------------------------------------------------
        # protein_group_id
        # ----------------------------------------------------------

        protein = check_array(

            "protein_group_id",

            file_maps["protein_group_id"][prefix],

            (None,),

            issues,

            dataset_name

        )

        if len(protein) != n:

            add_issue(
                issues,
                dataset_name,
                "protein_group_id",
                "FAIL",
                "Wrong protein_group_id length"
            )

        summary["protein_group_id_files"].append(

            file_maps["protein_group_id"][prefix]

        )

        # ----------------------------------------------------------
        # CSV
        # ----------------------------------------------------------

        check_csv(

            file_maps["metadata"][prefix],

            n,

            dataset_name,

            "metadata",

            issues

        )

        check_csv(

            file_maps["index"][prefix],

            n,

            dataset_name,

            "index",

            issues

        )

    # --------------------------------------------------------------
    # Chunk sizes
    # --------------------------------------------------------------

    sizes = summary["chunk_sizes"]

    if len(sizes) > 1:

        expected = Counter(

            sizes[:-1]

        ).most_common(1)[0][0]

        for i, size in enumerate(sizes[:-1]):

            if size != expected:

                add_issue(
                    issues,
                    dataset_name,
                    "chunks",
                    "WARNING",
                    f"Chunk {i} has {size} rows "
                    f"(expected {expected})"
                )

        if sizes[-1] > expected:

            add_issue(
                issues,
                dataset_name,
                "chunks",
                "WARNING",
                "Last chunk larger than expected"
            )

    return summary

# =============================================================================
# SECONDARY DATASET
# =============================================================================

def check_secondary_dataset(dataset_name,
                            secondary_dir,
                            issues):

    chunks_dir = Path(secondary_dir) / "chunks"

    summary = {

        "chunks": 0,
        "rows": 0,

        "group_id_files": [],
        "protein_group_id_files": [],

        "chunk_sizes": []

    }

    file_maps, prefixes = check_required_files(

        dataset_name,

        chunks_dir,

        SECONDARY_FILES,

        issues

    )

    summary["chunks"] = len(prefixes)

    missing, duplicated = check_chunk_sequence(
        prefixes
    )

    if missing:

        add_issue(
            issues,
            dataset_name,
            "secondary_chunks",
            "FAIL",
            f"Missing chunks {missing}"
        )

    if duplicated:

        add_issue(
            issues,
            dataset_name,
            "secondary_chunks",
            "FAIL",
            f"Duplicated chunks {duplicated}"
        )

    # ----------------------------------------------------------
    # Iterate over chunks
    # ----------------------------------------------------------

    for prefix in prefixes:

        x = check_array(

            "X_ss",

            file_maps["X_ss"][prefix],

            (None,
             SECONDARY_SHAPE[0],
             SECONDARY_SHAPE[1]),

            issues,

            dataset_name

        )

        n = x.shape[0]

        summary["rows"] += n

        summary["chunk_sizes"].append(
            n
        )

        group = check_array(

            "group_id",

            file_maps["group_id"][prefix],

            (None,),

            issues,

            dataset_name

        )

        if len(group) != n:

            add_issue(
                issues,
                dataset_name,
                "group_id",
                "FAIL",
                "Wrong group_id length"
            )

        summary["group_id_files"].append(

            file_maps["group_id"][prefix]

        )

        protein = check_array(

            "protein_group_id",

            file_maps["protein_group_id"][prefix],

            (None,),

            issues,

            dataset_name

        )

        if len(protein) != n:

            add_issue(
                issues,
                dataset_name,
                "protein_group_id",
                "FAIL",
                "Wrong protein_group_id length"
            )

        summary["protein_group_id_files"].append(

            file_maps["protein_group_id"][prefix]

        )

    # ----------------------------------------------------------
    # Chunk sizes
    # ----------------------------------------------------------

    sizes = summary["chunk_sizes"]

    if len(sizes) > 1:

        expected = Counter(
            sizes[:-1]
        ).most_common(1)[0][0]

        for i, size in enumerate(sizes[:-1]):

            if size != expected:

                add_issue(
                    issues,
                    dataset_name,
                    "secondary_chunks",
                    "WARNING",
                    f"Chunk {i} has {size} rows "
                    f"(expected {expected})"
                )

        if sizes[-1] > expected:

            add_issue(
                issues,
                dataset_name,
                "secondary_chunks",
                "WARNING",
                "Last chunk larger than expected"
            )

    # ----------------------------------------------------------
    # Global files
    # ----------------------------------------------------------

    for file in [

        "group_id.npy",

        "protein_group_id.npy",

        "index.csv",

        "metadata.csv"

    ]:

        path = Path(secondary_dir) / file

        if not path.exists():

            add_issue(

                issues,

                dataset_name,

                "secondary",

                "WARNING",

                f"Missing {file}"

            )

            continue

        if path.suffix == ".csv":

            check_csv(

                path,

                summary["rows"],

                dataset_name,

                file,

                issues

            )

        else:

            arr = np_load(path)

            if len(arr) != summary["rows"]:

                add_issue(
                    issues,
                    dataset_name,
                    file,
                    "FAIL",
                    "Wrong global array length"
                )

            if file == "group_id.npy":

                expected = concatenate(
                    summary["group_id_files"]
                )

            elif file == "protein_group_id.npy":

                expected = concatenate(
                    summary["protein_group_id_files"]
                )

            else:

                expected = None

            if expected is not None:

                if not np.array_equal(arr, expected):

                    add_issue(
                        issues,
                        dataset_name,
                        file,
                        "FAIL",
                        "Global array differs from chunk concatenation"
                    )

           
    return summary

# =============================================================================
# PRIMARY vs SECONDARY
# =============================================================================

def compare_primary_secondary(dataset_name,
                              primary,
                              secondary,
                              issues):
    """
    Compare primary and secondary IDs.
    """

    report = {}

    # ----------------------------------------------------------
    # group_id
    # ----------------------------------------------------------

    primary_group = concatenate(
        primary["group_id_files"]
    )

    secondary_group = concatenate(
        secondary["group_id_files"]
    )

    group = compare_ids(
        primary_group,
        secondary_group
    )

    if len(group["missing"]):

        add_issue(
            issues,
            dataset_name,
            "group_id",
            "WARNING",
            f"{len(group['missing'])} group_id missing in secondary"
        )

    if len(group["extra"]):

        add_issue(
            issues,
            dataset_name,
            "group_id",
            "FAIL",
            f"{len(group['extra'])} unexpected group_id in secondary"
        )

    report["group_id"] = group

    # ----------------------------------------------------------
    # protein_group_id
    # ----------------------------------------------------------

    primary_protein = concatenate(
        primary["protein_group_id_files"]
    )

    secondary_protein = concatenate(
        secondary["protein_group_id_files"]
    )

    protein = compare_ids(
        primary_protein,
        secondary_protein
    )

    if len(protein["missing"]):

        add_issue(
            issues,
            dataset_name,
            "protein_group_id",
            "WARNING",
            f"{len(protein['missing'])} proteins missing in secondary"
        )

    if len(protein["extra"]):

        add_issue(
            issues,
            dataset_name,
            "protein_group_id",
            "FAIL",
            f"{len(protein['extra'])} unexpected proteins in secondary"
        )

    report["protein_group_id"] = protein

    return report


# =============================================================================
# MAIN
# =============================================================================

def print_issues(issues):

    if not issues:
        return

    print("\nDetected issues:")

    for issue in issues:

        print(
            f"[{issue.level}] "
            f"{issue.dataset} | "
            f"{issue.section} | "
            f"{issue.message}"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Quality control for ProtT5 and SecondaryBERT embedding datasets."
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing all datasets."
    )

    args = parser.parse_args()

    datasets = discover_datasets(args.root)

    if not datasets:
        print("No datasets found.")
        return

    issues = []
    summaries = []

    print("=" * 80)
    print("Embedding Dataset Quality Control")
    print("=" * 80)

    for dataset in datasets:

        print(f"\nChecking: {dataset['name']}")

        primary = check_primary_dataset(
            dataset["name"],
            dataset["primary"],
            issues
        )

        secondary = check_secondary_dataset(
            dataset["name"],
            dataset["secondary"],
            issues
        )

        compare_primary_secondary(
            dataset["name"],
            primary,
            secondary,
            issues
        )
        summary = DatasetSummary(
            dataset=dataset["name"],
            primary_chunks=primary["chunks"],
            secondary_chunks=secondary["chunks"],
            primary_rows=primary["rows"],
            secondary_rows=secondary["rows"],
            positives=primary["labels"].get(1, 0),
            negatives=primary["labels"].get(0, 0)
        )

        dataset_issues = [
            i for i in issues
            if i.dataset == dataset["name"]
        ]

        if any(i.level == "FAIL" for i in dataset_issues):
            summary.result = "FAIL"

        elif any(i.level == "WARNING" for i in dataset_issues):
            summary.result = "PASS_WITH_WARNINGS"

        else:
            summary.result = "PASS"

        summaries.append(summary)

        print(f"  Primary chunks   : {summary.primary_chunks}")
        print(f"  Secondary chunks : {summary.secondary_chunks}")
        print(f"  Primary rows     : {summary.primary_rows}")
        print(f"  Secondary rows   : {summary.secondary_rows}")
        print(f"  Positives        : {summary.positives}")
        print(f"  Negatives        : {summary.negatives}")
        total = summary.positives + summary.negatives
        ratio = summary.positives / total if total else 0
        print(f"  Positive ratio   : {ratio:.2%}")
        print(f"  Result           : {summary.result}")

    print_issues(issues)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fails = sum(i.level == "FAIL" for i in issues)
    warnings = sum(i.level == "WARNING" for i in issues)

    print(f"Datasets checked : {len(summaries)}")
    print(f"FAIL             : {fails}")
    print(f"WARNING          : {warnings}")

    if fails:
        raise SystemExit(1)

    print("\nAll datasets successfully validated.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()