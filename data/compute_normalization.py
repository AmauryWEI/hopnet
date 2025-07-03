# File:         compute_normalization.py
# Date:         2024/05/07
# Description:  Top-level script to compute normalization parameters for a dataset

import argparse
import gzip
import os
from pathlib import Path
from sys import exit, path as spath

# Add the parent directory to the Python path
spath.append(os.path.join(Path(__file__).parent.resolve(), ".."))

import numpy as np
import torch
from tqdm import tqdm

# Environment Configuration
COMPLEXES_FILENAME: str = "complexes"
COMPRESSED_EXTENSION: str = ".npy.gz"


def compute_normalization(stats: list[tuple], nox4: bool) -> tuple:
    """Compute the normalization parameters from dataset statistics

    Parameters
    ----------
    stats : list[tuple]
        Gathered statistics for each feature and target matrix
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet

    Returns
    -------
    tuple
        Normalization mean and standard deviation vectors
    """
    x0_count = np.sum([s[0] for s in stats], axis=0)
    x0_sum = np.sum([s[1] for s in stats], axis=0)
    x0_sum_sq = np.sum([s[2] for s in stats], axis=0)

    x1_count = np.sum([s[3] for s in stats], axis=0)
    x1_sum = np.sum([s[4] for s in stats], axis=0)
    x1_sum_sq = np.sum([s[5] for s in stats], axis=0)

    x2_count = np.sum([s[6] for s in stats], axis=0)
    x2_sum = np.sum([s[7] for s in stats], axis=0)
    x2_sum_sq = np.sum([s[8] for s in stats], axis=0)

    x3_count = np.sum([s[9] for s in stats], axis=0)
    x3_sum = np.sum([s[10] for s in stats], axis=0)
    x3_sum_sq = np.sum([s[11] for s in stats], axis=0)

    t0_count = np.sum([s[15] for s in stats], axis=0)
    t0_sum = np.sum([s[16] for s in stats], axis=0)
    t0_sum_sq = np.sum([s[17] for s in stats], axis=0)

    x0_mean = x0_sum / x0_count
    x0_std = np.sqrt(x0_sum_sq / x0_count - x0_mean**2)
    x1_mean = x1_sum / x1_count
    x1_std = np.sqrt(x1_sum_sq / x1_count - x1_mean**2)
    x2_mean = x2_sum / x2_count
    x2_std = np.sqrt(x2_sum_sq / x2_count - x2_mean**2)
    x3_mean = x3_sum / x3_count
    x3_std = np.sqrt(x3_sum_sq / x3_count - x3_mean**2)
    t0_mean = t0_sum / t0_count
    t0_std = np.sqrt(t0_sum_sq / t0_count - t0_mean**2)

    # Fix standard deviations of 0 (to avoid division by 0 and reaching infinity)
    x0_idx = np.argwhere(np.abs(x0_std) < 1e-10)
    x0_std[x0_idx] = 1.0
    x1_idx = np.argwhere(np.abs(x1_std) < 1e-10)
    x1_std[x1_idx] = 1.0
    x2_idx = np.argwhere(np.abs(x2_std) < 1e-10)
    x2_std[x2_idx] = 1.0
    x3_idx = np.argwhere(np.abs(x3_std) < 1e-10)
    x3_std[x3_idx] = 1.0

    # Uniform scaling for input velocities (divide by the max velocity norm)
    x0_std[0:8] = np.max(x0_std[0:8])
    # Uniform scaling for input node distances to the object center mass
    x0_std[8:16] = np.max(x0_std[8:16])
    if nox4:
        # Equal normalization of obj_type
        x0_mean[16:18] = 0.5
        x0_std[16:18] = 0.5

    # Uniform scaling for face normal vectors
    x2_std[0:3] = np.max(x2_std)

    # Uniform scaling for output accelerations (divide by the max acceleration norm)
    t0_std[0:3] = np.max(t0_std)

    # Uniform scaling for collision distances (divide by the max distance)
    x3_std[0:4] = np.max(x3_std[0:4])  # [dist_x, dist_y, dist_z, dist_norm]
    # Uniform scaling for every 3D points
    x3_std[4:8] = np.max([x3_std[4:8], x3_std[16:20]])
    x3_std[16:20] = x3_std[4:8]
    x3_std[8:12] = np.max([x3_std[8:12], x3_std[20:24]])
    x3_std[20:24] = x3_std[8:12]
    x3_std[12:16] = np.max([x3_std[12:16], x3_std[24:28]])
    x3_std[24:28] = x3_std[12:16]

    # Uniform scaling for input distances (divide by the max distance)
    x1_std[:] = np.max(x1_std)

    if not nox4:
        x4_count = np.sum([s[12] for s in stats], axis=0)
        x4_sum = np.sum([s[13] for s in stats], axis=0)
        x4_sum_sq = np.sum([s[14] for s in stats], axis=0)
        x4_mean = x4_sum / x4_count
        x4_std = np.sqrt(x4_sum_sq / x4_count - x4_mean**2)

        t4_count = np.sum([s[18] for s in stats], axis=0)
        t4_sum = np.sum([s[19] for s in stats], axis=0)
        t4_sum_sq = np.sum([s[20] for s in stats], axis=0)
        t4_mean = t4_sum / t4_count
        t4_std = np.sqrt(t4_sum_sq / t4_count - t4_mean**2)

        # Uniform scaling for input velocities (divide by the max velocity norm)
        x4_std[0:8] = np.max(x4_std[0:8])
        # Equal normalization of obj_type
        x4_mean[8:10] = 0.5
        x4_std[8:10] = 0.5
        # Uniform scaling for output accelerations (divide by the max acceleration norm)
        t4_std[0:3] = np.max(t4_std)
    else:
        x4_mean = None
        x4_std = None
        t4_mean = None
        t4_std = None

    return (
        x0_mean,
        x0_std,
        x1_mean,
        x1_std,
        x2_mean,
        x2_std,
        x3_mean,
        x3_std,
        x4_mean,
        x4_std,
        t0_mean,
        t0_std,
        t4_mean,
        t4_std,
    )


def get_statistics(sample: str, filename: str) -> tuple:
    """Get statistics for one sample in a dataset

    Parameters
    ----------
    sample : str
        Path to the sample directory
    filename : str
        Filename containing the combinatorial complexes information

    Returns
    -------
    tuple
        Statistics about all learning features and learning targets matrices
    """
    f = gzip.GzipFile(os.path.join(sample, filename))
    data = np.load(f, allow_pickle=True).item()
    f.close()

    # Get the raw data for every timestep of the sample
    x0: torch.Tensor = data["x0"].numpy()
    x1: torch.Tensor = data["x1"].numpy()
    x4: torch.Tensor = data["x4"].numpy()
    t0: torch.Tensor = data["t0"].numpy()
    t4: torch.Tensor = data["t4"].numpy()

    x0_count = x0.shape[0] * x0.shape[1]  # Time * nodes_count
    x1_count = x1.shape[0] * x1.shape[1]  # Time * edges_count
    x4_count = x4.shape[0] * x4.shape[1]  # Time * obj_count

    # Manually count x2 and x4 (non-consistent shape throughout the experiment)
    x2_count = np.sum([x2.shape[0] if x2 is not None else 0 for x2 in data["x2_l"]])
    x3_count = np.sum([x3.shape[0] if x3 is not None else 0 for x3 in data["x3_l"]])

    # Compute the sum of the samples
    x0_sum = np.sum(x0, axis=(0, 1))
    x1_sum = np.sum(x1, axis=(0, 1))
    x4_sum = np.sum(x4, axis=(0, 1))
    t0_sum = np.sum(t0, axis=(0, 1))
    t4_sum = np.sum(t4, axis=(0, 1))

    # Compute the sum of the squared samples
    x0_sum_sq = np.sum(x0**2, axis=(0, 1))
    x1_sum_sq = np.sum(x1**2, axis=(0, 1))
    x4_sum_sq = np.sum(x4**2, axis=(0, 1))
    t0_sum_sq = np.sum(t0**2, axis=(0, 1))
    t4_sum_sq = np.sum(t4**2, axis=(0, 1))

    # Manually compute the sums for x2 and x4
    x2_sum = np.zeros(3)
    x2_sum_sq = np.zeros(3)
    for x2 in data["x2_l"]:
        if x2 is None:
            continue
        x2_sum += np.sum(x2.numpy(), axis=0)
        x2_sum_sq += np.sum(x2.numpy() ** 2, axis=0)

    x3_sum = np.zeros(28)
    x3_sum_sq = np.zeros(28)
    for x3 in data["x3_l"]:
        if x3 is None:
            continue
        x3_sum += np.sum(x3.numpy(), axis=0)
        x3_sum_sq += np.sum(x3.numpy() ** 2, axis=0)

    return (
        np.atleast_1d(np.array(x0_count)),
        x0_sum,
        x0_sum_sq,
        np.atleast_1d(np.array(x1_count)),
        x1_sum,
        x1_sum_sq,
        np.atleast_1d(np.array(x2_count)),
        x2_sum,
        x2_sum_sq,
        np.atleast_1d(np.array(x3_count)),
        x3_sum,
        x3_sum_sq,
        np.atleast_1d(np.array(x4_count)),
        x4_sum,
        x4_sum_sq,
        np.atleast_1d(np.array(x0_count)),
        t0_sum,
        t0_sum_sq,
        np.atleast_1d(np.array(x4_count)),
        t4_sum,
        t4_sum_sq,
    )


def main(args: argparse.Namespace) -> int:
    # Make sure the dataset exists
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset dir {dataset_dir} does not exist.")

    # List the samples in the dataset
    samples = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
    try:
        seeds = set([int(f) for f in samples if len(f) <= 4])
        print(f"INFO: Found {len(seeds)} real samples in dataset directory.")
    except Exception as e:
        print(f"ERROR: Dataset directory contains non-digit folders.")
        print("\t=>", e)
        return 1

    samples_full = [os.path.join(dataset_dir, s) for s in samples if len(s) <= 4]
    empty_samples = []
    filename = (
        COMPLEXES_FILENAME + (f"-nox4" if args.nox4 else "") + COMPRESSED_EXTENSION
    )

    for exp in samples_full:
        f = os.path.join(exp, filename)
        if not (os.path.isfile(f) and os.stat(f).st_size > 0):
            empty_samples.append(exp)
    [samples_full.remove(a) for a in empty_samples]
    print(f"INFO: Found {len(empty_samples)} empty experiments.")

    if len(samples_full) == 0:
        print(f"ERROR: No samples found in {dataset_dir}.")
        return 1

    samples_full = sorted(
        samples_full, key=lambda f: int(os.path.basename(os.path.normpath(f)))
    )

    # For each experiment, get its statistics
    stats: list[tuple] = []
    for exp in tqdm(samples_full):
        stats.append(get_statistics(exp, filename))

    # Combine all experiments data into mean and std
    statistics = compute_normalization(stats, args.nox4)

    (
        x0_mean,
        x0_std,
        x1_mean,
        x1_std,
        x2_mean,
        x2_std,
        x3_mean,
        x3_std,
        x4_mean,
        x4_std,
        t0_mean,
        t0_std,
        t4_mean,
        t4_std,
    ) = statistics

    # Save the data
    np.save(
        os.path.join(args.out_dir, "normalization" + ("-nox4" if args.nox4 else "")),
        {
            "x0_mean": x0_mean,
            "x0_std": x0_std,
            "x1_mean": x1_mean,
            "x1_std": x1_std,
            "x2_mean": x2_mean,
            "x2_std": x2_std,
            "x3_mean": x3_mean,
            "x3_std": x3_std,
            "x4_mean": x4_mean,
            "x4_std": x4_std,
            "t0_mean": t0_mean,
            "t0_std": t0_std,
            "t4_mean": t4_mean,
            "t4_std": t4_std,
        },  # type: ignore
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Normalization Parameters")
    parser.add_argument("dataset_dir", type=str, help="Location of the MoVi dataset")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory")
    parser.add_argument(
        "--nox4",
        action="store_true",
        default=False,
        help="For ablation study without object cells X4",
    )
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
