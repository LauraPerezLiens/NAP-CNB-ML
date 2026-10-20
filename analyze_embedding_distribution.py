#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Embedding Distribution Analysis
===============================================================================

Analyze the numerical distribution of:

    - PRIMARY: ProtT5 embeddings
    - SECONDARY: mean-pooled SecondaryBERT embeddings

The analysis is restricted to the 11 expected NAP-CNB datasets.

The script samples rows from several chunks instead of loading complete datasets
into RAM.

For each modality it reports:

    - embedding dimension
    - number of sampled rows / values
    - min / max
    - mean / standard deviation
    - percentiles
    - NaN / Inf counts
    - per-dimension mean/std statistics
    - near-zero variance dimensions
    - stability of feature mean/std estimates using 1, 3 and 5 chunks

IMPORTANT
---------
Current SecondaryBERT embeddings are already mean-pooled and have shape:

    (N, 768)

Therefore they are read directly from:

    *_SecondaryBERT_by_group/chunks/chunk_XXXXXX_X_ss.npy

No chunks_pooled directory is used.

No training data are modified.
===============================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PRIMARY_PATTERN = "chunk_*_X.npy"
SECONDARY_PATTERN = "chunk_*_X_ss.npy"

EXPECTED_PRIMARY_DIM = 1024
EXPECTED_SECONDARY_DIM = 768

DEFAULT_N_CHUNKS = 5
DEFAULT_ROWS_PER_CHUNK = 10_000
DEFAULT_RANDOM_STATE = 42

NEAR_ZERO_STD = 1e-8


EXPECTED_DATASETS = [
    ("human", "mhc-I",  "HLA-A"),
    ("human", "mhc-I",  "HLA-B"),
    ("human", "mhc-I",  "HLA-C"),

    ("human", "mhc-II", "HLA-DP"),
    ("human", "mhc-II", "HLA-DQ"),
    ("human", "mhc-II", "HLA-DR"),

    ("mouse", "mhc-I",  "H2-D"),
    ("mouse", "mhc-I",  "H2-K"),
    ("mouse", "mhc-I",  "H2-L"),

    ("mouse", "mhc-II", "H2-IA"),
    ("mouse", "mhc-II", "H2-IE"),
]


PERCENTILES = [
    0.001,
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
    0.999,
]


# =============================================================================
# Containers
# =============================================================================

@dataclass
class ModalityAnalysis:
    name: str

    embedding_dim: int

    sampled_rows: int
    sampled_values: int

    minimum: float
    maximum: float

    mean: float
    std: float

    nan_count: int
    inf_count: int

    percentiles: Dict[str, float]

    feature_means: np.ndarray
    feature_stds: np.ndarray


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze distributions of ProtT5 and "
            "mean-pooled SecondaryBERT embeddings."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing data_input_nn datasets. "
            "Example: /data/nap/lperez_nn/data/data_input_nn"
        ),
    )

    parser.add_argument(
        "--n-chunks",
        type=int,
        default=DEFAULT_N_CHUNKS,
        help=(
            "Maximum number of chunks sampled "
            "per modality/dataset."
        ),
    )

    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=DEFAULT_ROWS_PER_CHUNK,
        help=(
            "Maximum number of randomly sampled rows "
            "from each chunk."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    parser.add_argument(
        "--output",
        default="embedding_distribution_analysis",
        help=(
            "Directory where CSV summaries "
            "will be written."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Dataset discovery
# =============================================================================

def find_primary_directory(
    dataset_root: Path,
) -> Optional[Path]:
    """
    Find the primary ProtT5 chunk directory for exactly one dataset.

    We prefer known current naming conventions but retain a conservative
    fallback based on directories that actually contain chunk_*_X.npy.
    """

    # -------------------------------------------------------------------------
    # First try expected naming conventions.
    # -------------------------------------------------------------------------

    preferred_patterns = [
        "*_blosum_ProtT5_chunks",
        "*_ProtT5_chunks",
    ]

    preferred_matches = []

    for pattern in preferred_patterns:

        for path in dataset_root.glob(pattern):

            if (
                path.is_dir()
                and any(path.glob(PRIMARY_PATTERN))
            ):
                preferred_matches.append(path)

    preferred_matches = sorted(
        set(preferred_matches)
    )

    if len(preferred_matches) == 1:
        return preferred_matches[0]

    if len(preferred_matches) > 1:

        raise RuntimeError(
            "Multiple possible primary directories found under "
            f"{dataset_root}:\n"
            + "\n".join(
                f"    {path}"
                for path in preferred_matches
            )
        )

    # -------------------------------------------------------------------------
    # Conservative fallback.
    # -------------------------------------------------------------------------

    candidates = set()

    for chunk_file in dataset_root.rglob(
        PRIMARY_PATTERN
    ):
        candidates.add(
            chunk_file.parent
        )

    candidates = sorted(candidates)

    # Do not accidentally consider SecondaryBERT directories primary.
    candidates = [
        path
        for path in candidates
        if "SecondaryBERT" not in path.name
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None

    raise RuntimeError(
        "Multiple possible primary directories found under "
        f"{dataset_root}:\n"
        + "\n".join(
            f"    {path}"
            for path in candidates
        )
    )


def find_secondary_directory(
    dataset_root: Path,
) -> Optional[Path]:
    """
    Find the current SecondaryBERT_by_group directory.
    """

    candidates = sorted(
        path
        for path in dataset_root.glob(
            "*_SecondaryBERT_by_group"
        )
        if path.is_dir()
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None

    raise RuntimeError(
        "Multiple SecondaryBERT directories found under "
        f"{dataset_root}:\n"
        + "\n".join(
            f"    {path}"
            for path in candidates
        )
    )


# =============================================================================
# Sampling
# =============================================================================

def choose_chunks(
    chunk_files: Sequence[Path],
    n_chunks: int,
    rng: np.random.Generator,
) -> List[Path]:
    """
    Randomly choose up to n_chunks without replacement.
    """

    if not chunk_files:
        return []

    n_select = min(
        n_chunks,
        len(chunk_files),
    )

    indices = rng.choice(
        len(chunk_files),
        size=n_select,
        replace=False,
    )

    indices = np.sort(indices)

    return [
        chunk_files[int(index)]
        for index in indices
    ]


def sample_chunk_rows(
    chunk_file: Path,
    rows_per_chunk: int,
    rng: np.random.Generator,
    expected_dim: int,
) -> np.ndarray:
    """
    Memory-map one chunk and materialize only a random subset of rows.
    """

    values = np.load(
        chunk_file,
        mmap_mode="r",
    )

    if values.ndim != 2:

        raise RuntimeError(
            f"Expected a 2-D embedding array in {chunk_file}, "
            f"found shape {values.shape}."
        )

    if values.shape[1] != expected_dim:

        raise RuntimeError(
            f"Unexpected embedding dimension in {chunk_file}: "
            f"expected {expected_dim}, found {values.shape[1]}."
        )

    n_rows = len(values)

    if n_rows == 0:

        raise RuntimeError(
            f"Empty embedding chunk: {chunk_file}"
        )

    n_select = min(
        rows_per_chunk,
        n_rows,
    )

    row_indices = rng.choice(
        n_rows,
        size=n_select,
        replace=False,
    )

    # Sorting random indices generally improves mmap access locality.
    row_indices.sort()

    sample = np.asarray(
        values[row_indices],
        dtype=np.float32,
    )

    return sample


def collect_samples(
    chunk_files: Sequence[Path],
    rows_per_chunk: int,
    rng: np.random.Generator,
    expected_dim: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Sample rows from selected chunks.
    """

    samples: List[np.ndarray] = []

    for chunk_number, chunk_file in enumerate(
        chunk_files,
        start=1,
    ):

        values = np.load(
            chunk_file,
            mmap_mode="r",
        )

        print(
            f"      [{chunk_number}/{len(chunk_files)}] "
            f"{chunk_file.name}: "
            f"{values.shape} {values.dtype}"
        )

        if values.ndim != 2:

            raise RuntimeError(
                f"Expected 2-D embeddings in {chunk_file}; "
                f"found {values.shape}."
            )

        if values.shape[1] != expected_dim:

            raise RuntimeError(
                f"Embedding dimension mismatch in {chunk_file}: "
                f"expected {expected_dim}, "
                f"found {values.shape[1]}."
            )

        sample = sample_chunk_rows(
            chunk_file,
            rows_per_chunk,
            rng,
            expected_dim,
        )

        samples.append(sample)

    if not samples:

        raise RuntimeError(
            "No embedding samples were collected."
        )

    combined = np.concatenate(
        samples,
        axis=0,
    )

    return combined, samples


# =============================================================================
# Statistics
# =============================================================================

def analyze_values(
    name: str,
    values: np.ndarray,
) -> ModalityAnalysis:

    if values.ndim != 2:

        raise ValueError(
            f"{name}: expected 2-D matrix, "
            f"found {values.shape}."
        )

    nan_count = int(
        np.count_nonzero(
            np.isnan(values)
        )
    )

    inf_count = int(
        np.count_nonzero(
            np.isinf(values)
        )
    )

    if nan_count or inf_count:

        raise RuntimeError(
            f"{name} contains non-finite values: "
            f"NaN={nan_count:,}, "
            f"Inf={inf_count:,}"
        )

    flat = values.reshape(-1)

    percentile_values = np.quantile(
        flat,
        PERCENTILES,
    )

    percentile_dict = {
        f"p{100 * percentile:g}": float(value)
        for percentile, value in zip(
            PERCENTILES,
            percentile_values,
        )
    }

    feature_means = values.mean(
        axis=0,
        dtype=np.float64,
    )

    feature_stds = values.std(
        axis=0,
        dtype=np.float64,
    )

    return ModalityAnalysis(
        name=name,

        embedding_dim=values.shape[1],

        sampled_rows=values.shape[0],
        sampled_values=values.size,

        minimum=float(values.min()),
        maximum=float(values.max()),

        mean=float(
            values.mean(dtype=np.float64)
        ),

        std=float(
            values.std(dtype=np.float64)
        ),

        nan_count=nan_count,
        inf_count=inf_count,

        percentiles=percentile_dict,

        feature_means=feature_means,
        feature_stds=feature_stds,
    )


def print_analysis(
    analysis: ModalityAnalysis,
) -> None:

    print()
    print(
        f"    {analysis.name.upper()}"
    )

    print(
        "    " + "-" * 70
    )

    print(
        f"      embedding dimension : "
        f"{analysis.embedding_dim:,}"
    )

    print(
        f"      sampled rows        : "
        f"{analysis.sampled_rows:,}"
    )

    print(
        f"      sampled values      : "
        f"{analysis.sampled_values:,}"
    )

    print()
    print(
        "      GLOBAL DISTRIBUTION"
    )

    print(
        f"        min               : "
        f"{analysis.minimum:.6f}"
    )

    print(
        f"        max               : "
        f"{analysis.maximum:.6f}"
    )

    print(
        f"        mean              : "
        f"{analysis.mean:.6f}"
    )

    print(
        f"        std               : "
        f"{analysis.std:.6f}"
    )

    for name, value in (
        analysis.percentiles.items()
    ):

        print(
            f"        {name:<17}: "
            f"{value:.6f}"
        )

    print()
    print(
        "      PER-DIMENSION MEANS"
    )

    print(
        "        min(mean_j)       : "
        f"{analysis.feature_means.min():.6f}"
    )

    print(
        "        max(mean_j)       : "
        f"{analysis.feature_means.max():.6f}"
    )

    print(
        "        median(mean_j)    : "
        f"{np.median(analysis.feature_means):.6f}"
    )

    print()
    print(
        "      PER-DIMENSION STANDARD DEVIATIONS"
    )

    print(
        "        min(std_j)        : "
        f"{analysis.feature_stds.min():.6f}"
    )

    print(
        "        max(std_j)        : "
        f"{analysis.feature_stds.max():.6f}"
    )

    print(
        "        median(std_j)     : "
        f"{np.median(analysis.feature_stds):.6f}"
    )

    near_zero = int(
        np.count_nonzero(
            analysis.feature_stds
            < NEAR_ZERO_STD
        )
    )

    print(
        "        near-zero std     : "
        f"{near_zero}/"
        f"{analysis.embedding_dim}"
    )

    print()

    print(
        f"      NaN               : "
        f"{analysis.nan_count:,}"
    )

    print(
        f"      Inf               : "
        f"{analysis.inf_count:,}"
    )


# =============================================================================
# Stability analysis
# =============================================================================

def calculate_feature_statistics(
    samples: Sequence[np.ndarray],
    n_chunks: int,
) -> Tuple[np.ndarray, np.ndarray]:

    selected = np.concatenate(
        samples[:n_chunks],
        axis=0,
    )

    means = selected.mean(
        axis=0,
        dtype=np.float64,
    )

    stds = selected.std(
        axis=0,
        dtype=np.float64,
    )

    return means, stds


def analyze_stability(
    modality_name: str,
    samples: Sequence[np.ndarray],
) -> List[Dict[str, float]]:

    print()
    print(
        f"    ESTIMATE STABILITY — "
        f"{modality_name.upper()}"
    )

    print(
        "    " + "-" * 70
    )

    available = len(samples)

    checkpoints = [
        number
        for number in (1, 3, 5)
        if number <= available
    ]

    if available not in checkpoints:
        checkpoints.append(available)

    checkpoints = sorted(
        set(checkpoints)
    )

    reference_n = max(checkpoints)

    reference_mean, reference_std = (
        calculate_feature_statistics(
            samples,
            reference_n,
        )
    )

    rows = []

    print(
        f"      Reference: "
        f"{reference_n} chunks"
    )

    print()

    print(
        "      chunks    "
        "mean MAE       "
        "mean max diff    "
        "std MAE        "
        "std max diff"
    )

    for n_chunks in checkpoints:

        means, stds = (
            calculate_feature_statistics(
                samples,
                n_chunks,
            )
        )

        mean_diff = np.abs(
            means - reference_mean
        )

        std_diff = np.abs(
            stds - reference_std
        )

        row = {
            "modality": modality_name,
            "n_chunks": n_chunks,

            "mean_mae_vs_reference":
                float(mean_diff.mean()),

            "mean_max_diff_vs_reference":
                float(mean_diff.max()),

            "std_mae_vs_reference":
                float(std_diff.mean()),

            "std_max_diff_vs_reference":
                float(std_diff.max()),
        }

        rows.append(row)

        print(
            f"      {n_chunks:<9}"
            f"{row['mean_mae_vs_reference']:<15.8f}"
            f"{row['mean_max_diff_vs_reference']:<17.8f}"
            f"{row['std_mae_vs_reference']:<15.8f}"
            f"{row['std_max_diff_vs_reference']:.8f}"
        )

    return rows


# =============================================================================
# Output helpers
# =============================================================================

def analysis_to_summary_row(
    dataset_name: str,
    analysis: ModalityAnalysis,
) -> Dict[str, object]:

    row: Dict[str, object] = {
        "dataset": dataset_name,
        "modality": analysis.name,

        "embedding_dim":
            analysis.embedding_dim,

        "sampled_rows":
            analysis.sampled_rows,

        "sampled_values":
            analysis.sampled_values,

        "min":
            analysis.minimum,

        "max":
            analysis.maximum,

        "mean":
            analysis.mean,

        "std":
            analysis.std,

        "nan_count":
            analysis.nan_count,

        "inf_count":
            analysis.inf_count,

        "feature_mean_min":
            float(
                analysis.feature_means.min()
            ),

        "feature_mean_max":
            float(
                analysis.feature_means.max()
            ),

        "feature_mean_median":
            float(
                np.median(
                    analysis.feature_means
                )
            ),

        "feature_std_min":
            float(
                analysis.feature_stds.min()
            ),

        "feature_std_max":
            float(
                analysis.feature_stds.max()
            ),

        "feature_std_median":
            float(
                np.median(
                    analysis.feature_stds
                )
            ),

        "near_zero_std_dimensions":
            int(
                np.count_nonzero(
                    analysis.feature_stds
                    < NEAR_ZERO_STD
                )
            ),
    }

    row.update(
        analysis.percentiles
    )

    return row


def save_feature_statistics(
    output_dir: Path,
    dataset_name: str,
    analysis: ModalityAnalysis,
) -> None:

    safe_name = (
        dataset_name
        .replace("/", "_")
        .replace("\\", "_")
    )

    frame = pd.DataFrame(
        {
            "dimension": np.arange(
                analysis.embedding_dim
            ),

            "mean":
                analysis.feature_means,

            "std":
                analysis.feature_stds,
        }
    )

    frame.to_csv(
        output_dir
        / (
            f"{safe_name}_"
            f"{analysis.name}_"
            "per_dimension.csv"
        ),
        index=False,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    if args.n_chunks < 1:

        raise ValueError(
            "--n-chunks must be >= 1."
        )

    if args.rows_per_chunk < 1:

        raise ValueError(
            "--rows-per-chunk must be >= 1."
        )

    root = Path(
        args.root
    ).expanduser().resolve()

    output_dir = Path(
        args.output
    ).expanduser().resolve()

    if not root.is_dir():

        raise FileNotFoundError(
            f"Root directory does not exist: "
            f"{root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "EMBEDDING DISTRIBUTION ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"Root:             {root}"
    )

    print(
        f"Expected datasets:{len(EXPECTED_DATASETS):>4}"
    )

    print(
        f"Chunks/dataset:   {args.n_chunks}"
    )

    print(
        f"Rows/chunk:       "
        f"{args.rows_per_chunk:,}"
    )

    print(
        f"Random state:     "
        f"{args.random_state}"
    )

    print(
        f"Primary dim:      "
        f"{EXPECTED_PRIMARY_DIM}"
    )

    print(
        f"Secondary dim:    "
        f"{EXPECTED_SECONDARY_DIM}"
    )

    print(
        "=" * 100
    )

    summary_rows = []
    stability_rows = []

    successful_datasets = 0

    # =========================================================================
    # Iterate exactly over expected datasets
    # =========================================================================

    for dataset_number, (
        species,
        mhc,
        allele,
    ) in enumerate(
        EXPECTED_DATASETS,
        start=1,
    ):

        dataset_root = (
            root
            / species
            / mhc
            / allele
        )

        dataset_name = (
            f"{species}/{mhc}/{allele}"
        )

        print()
        print(
            "=" * 100
        )

        print(
            f"DATASET "
            f"[{dataset_number}/"
            f"{len(EXPECTED_DATASETS)}]"
        )

        print(
            dataset_name
        )

        print(
            "=" * 100
        )

        if not dataset_root.is_dir():

            raise FileNotFoundError(
                "Expected dataset directory "
                f"does not exist: {dataset_root}"
            )

        # =====================================================================
        # Locate PRIMARY and SECONDARY
        # =====================================================================

        primary_dir = (
            find_primary_directory(
                dataset_root
            )
        )

        secondary_dir = (
            find_secondary_directory(
                dataset_root
            )
        )

        if primary_dir is None:

            raise FileNotFoundError(
                "No primary ProtT5 directory "
                f"found for {dataset_name}"
            )

        if secondary_dir is None:

            raise FileNotFoundError(
                "No SecondaryBERT_by_group "
                f"directory found for "
                f"{dataset_name}"
            )

        print(
            f"  Primary directory:"
        )
        print(
            f"    {primary_dir}"
        )

        print(
            f"  Secondary directory:"
        )
        print(
            f"    {secondary_dir}"
        )

        # =====================================================================
        # PRIMARY — ProtT5
        # =====================================================================

        primary_files = sorted(
            primary_dir.glob(
                PRIMARY_PATTERN
            )
        )

        if not primary_files:

            raise RuntimeError(
                "No primary ProtT5 chunks "
                f"found in {primary_dir}"
            )

        primary_rng = (
            np.random.default_rng(
                args.random_state
                + dataset_number
            )
        )

        selected_primary = (
            choose_chunks(
                primary_files,
                args.n_chunks,
                primary_rng,
            )
        )

        print()
        print(
            "  PRIMARY — ProtT5"
        )

        print(
            f"    Available chunks: "
            f"{len(primary_files)}"
        )

        print(
            f"    Selected chunks:  "
            f"{len(selected_primary)}"
        )

        primary_values, primary_samples = (
            collect_samples(
                selected_primary,
                args.rows_per_chunk,
                primary_rng,
                EXPECTED_PRIMARY_DIM,
            )
        )

        primary_analysis = (
            analyze_values(
                "primary",
                primary_values,
            )
        )

        print_analysis(
            primary_analysis
        )

        primary_stability = (
            analyze_stability(
                "primary",
                primary_samples,
            )
        )

        for row in primary_stability:
            row["dataset"] = dataset_name

        stability_rows.extend(
            primary_stability
        )

        summary_rows.append(
            analysis_to_summary_row(
                dataset_name,
                primary_analysis,
            )
        )

        save_feature_statistics(
            output_dir,
            dataset_name,
            primary_analysis,
        )

        # Free memory before loading SecondaryBERT.
        del primary_values
        del primary_samples

        # =====================================================================
        # SECONDARY — SecondaryBERT
        # =====================================================================

        # IMPORTANT:
        # Current embeddings are already mean-pooled.
        # Read DIRECTLY from chunks/, NOT chunks_pooled/.
        secondary_chunks_dir = (
            secondary_dir
            / "chunks"
        )

        if not secondary_chunks_dir.is_dir():

            raise FileNotFoundError(
                "SecondaryBERT chunks directory "
                f"not found: "
                f"{secondary_chunks_dir}"
            )

        secondary_files = sorted(
            secondary_chunks_dir.glob(
                SECONDARY_PATTERN
            )
        )

        if not secondary_files:

            raise RuntimeError(
                "No SecondaryBERT chunks "
                f"found in "
                f"{secondary_chunks_dir}"
            )

        secondary_rng = (
            np.random.default_rng(
                args.random_state
                + 100_000
                + dataset_number
            )
        )

        selected_secondary = (
            choose_chunks(
                secondary_files,
                args.n_chunks,
                secondary_rng,
            )
        )

        print()
        print(
            "  SECONDARY — "
            "SecondaryBERT mean-pooled"
        )

        print(
            f"    Available chunks: "
            f"{len(secondary_files)}"
        )

        print(
            f"    Selected chunks:  "
            f"{len(selected_secondary)}"
        )

        secondary_values, secondary_samples = (
            collect_samples(
                selected_secondary,
                args.rows_per_chunk,
                secondary_rng,
                EXPECTED_SECONDARY_DIM,
            )
        )

        secondary_analysis = (
            analyze_values(
                "secondary",
                secondary_values,
            )
        )

        print_analysis(
            secondary_analysis
        )

        secondary_stability = (
            analyze_stability(
                "secondary",
                secondary_samples,
            )
        )

        for row in secondary_stability:
            row["dataset"] = dataset_name

        stability_rows.extend(
            secondary_stability
        )

        summary_rows.append(
            analysis_to_summary_row(
                dataset_name,
                secondary_analysis,
            )
        )

        save_feature_statistics(
            output_dir,
            dataset_name,
            secondary_analysis,
        )

        del secondary_values
        del secondary_samples

        successful_datasets += 1

    # =========================================================================
    # Save global summaries
    # =========================================================================

    summary_frame = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir
        / "embedding_distribution_summary.csv"
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    stability_frame = pd.DataFrame(
        stability_rows
    )

    stability_path = (
        output_dir
        / "embedding_scaler_stability.csv"
    )

    stability_frame.to_csv(
        stability_path,
        index=False,
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "ANALYSIS COMPLETED"
    )

    print(
        "=" * 100
    )

    print(
        f"Datasets analyzed: "
        f"{successful_datasets}/"
        f"{len(EXPECTED_DATASETS)}"
    )

    print(
        f"Summary:   "
        f"{summary_path}"
    )

    print(
        f"Stability: "
        f"{stability_path}"
    )

    print(
        "Per-dimension statistics: "
        f"{output_dir}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()