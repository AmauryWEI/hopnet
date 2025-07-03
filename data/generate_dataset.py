"""
Main script to generate the Multi-Object Video (MOVi) dataset A.
Objects:
  * The number of objects is randomly chosen between
    --min_num_objects (2) and --max_num_objects (5)
  * The objects are randomly chosen from the CLEVR objects set.
  * They are either rubber or metallic with different different colors and sizes

MOVid-A
  --camera=clevr --background=clevr --objects_set=clevr
  --min_num_objects=3 --max_num_objects=10
"""

import argparse
from functools import partial
from multiprocessing.pool import Pool
import os
import subprocess
import shutil
import sys

from tqdm import tqdm

SCRIPT_DIR: str = os.path.dirname(os.path.realpath(__file__))


# Simulator and Dataset Configuration
FRAME_RATE_HZ: int = 240  # Frequency of poses (and camera frames) in each experiment
DURATION_SEC: int = 2  # Duration of a single experiment
STEP_RATE_HZ: int = 5 * 240  # Simulation step rate (>= FRAME_RATE_HZ)
RESOLUTION_PX: int = 256

# Derived Constants
FRAME_START: int = 0
FRAME_END: int = FRAME_RATE_HZ * DURATION_SEC - 1


def gen_experiment(seed: int, out_dir: str, args: argparse.Namespace) -> None:
    # Create a unique directory for this experiment
    experiment_dir = os.path.join(out_dir, str(seed))
    os.makedirs(experiment_dir)

    if args.movis:
        camera = "clevr"
        background = "clevr"
        objects_set = "clevr"
    elif args.movia:
        camera = "clevr"
        background = "clevr"
        objects_set = "spheres"
    else:
        camera = "random"
        background = "colored"
        objects_set = "kubasic"

    # Generate the experiment using the movi_worker.py script
    subprocess.run(
        [
            "python",
            f"{SCRIPT_DIR}/movi_worker.py",
            f"--camera={camera}",
            f"--background={background}",
            "--rgb_only",
            f"--resolution={RESOLUTION_PX}",
            f"--frame_rate={FRAME_RATE_HZ}",
            f"--frame_start={FRAME_START}",
            f"--frame_end={FRAME_END}",
            f"--step_rate={STEP_RATE_HZ}",
            f"--seed={seed}",
            f"--job_dir={experiment_dir}",
            f"--min_num_objects={args.min_num_objects}",
            f"--max_num_objects={args.max_num_objects}",
            f"--objects_set={objects_set}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Create the MP4 video of the RGB images
    subprocess.run(
        [
            "ffmpeg",
            "-r",
            f"{FRAME_RATE_HZ}",
            "-f",
            "image2",
            "-i",
            f"{experiment_dir}/rgba_%05d.png",
            "-vcodec",
            "libx265",
            "-preset",
            "veryslow",
            "-crf",
            "20",
            f"{experiment_dir}/rgba.mp4",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Remove the raw PNG files
    png_files = [f for f in os.listdir(experiment_dir) if f.endswith(".png")]
    for f in png_files:
        os.remove(os.path.join(experiment_dir, f))


def main(args: argparse.Namespace) -> int:
    # Check if the starting seed is non-zero (as zero = random seed)
    if args.starting_seed == 0:
        print("ERROR: Starting seed must be > 0.")
        return 1

    if not args.movis and not args.movia and not args.movib:
        print("ERROR: A dataset configuration must be specified.")
        return 1

    if args.movis and args.out_dir == "":
        args.out_dir = "MoVi-spheres"
    elif args.movia and args.out_dir == "":
        args.out_dir = "MoVi-A"
    elif args.movib and args.out_dir == "":
        args.out_dir = "MoVi-B"

    # Check if ffmpeg is installed (required to create the videos)
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg must be installed and available in the PATH.")
        return 1

    requested_seeds = range(args.starting_seed, args.starting_seed + args.experiments)
    existing_seeds: set = set()

    # Check if the output directory is empty or not
    out_dir = os.path.abspath(args.out_dir)
    if os.path.isdir(out_dir):
        # If there are already experiments, check their seed
        print(f"INFO: Output directory already exists.")
        existing_exp = [f.name for f in os.scandir(out_dir) if f.is_dir()]
        try:
            existing_seeds = set([int(f) for f in existing_exp])
            print(f"INFO: Found {len(existing_seeds)} experiments in output directory.")
        except Exception as e:
            print(f"ERROR: Output directory contains non-digit experiment IDs")
            print("\t=>", e)
            return 1

        # Remove empty directories tackled by this job (happens if stopped with SIGTERM)
        for dir in existing_exp:
            if int(dir) not in requested_seeds:
                continue
            with os.scandir(os.path.join(out_dir, dir)) as it:
                if not any(it):
                    shutil.rmtree(os.path.join(out_dir, dir))
                    existing_seeds.remove(int(dir))
        pruned_dirs_num: int = len(existing_exp) - len(existing_seeds)
        if pruned_dirs_num:
            print(f"INFO: Pruned {pruned_dirs_num} empty directory(ies).")
    else:
        # Create the output directory
        os.makedirs(out_dir)
        print(f"INFO: Created output directory.")

    requested_seeds = range(args.starting_seed, args.starting_seed + args.experiments)
    seeds_to_compute: set = set(requested_seeds) - existing_seeds

    if len(seeds_to_compute) == 0:
        print(f"INFO: All experiments already exist in the output directory.")
        return 0

    print(f"INFO: Computing experiments for {len(seeds_to_compute)} seed(s) total.")
    with Pool(processes=args.threads) as pool:
        r = list(
            tqdm(
                pool.imap(
                    partial(
                        gen_experiment,
                        out_dir=out_dir,
                        args=args,
                    ),
                    seeds_to_compute,
                ),
                total=len(seeds_to_compute),
            )
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_num_objects", type=int, default=3, help="Minimum number of objects"
    )
    parser.add_argument(
        "--max_num_objects", type=int, default=10, help="Maximum number of objects"
    )
    parser.add_argument(
        "--experiments", type=int, default=1200, help="Total number of experiments"
    )
    parser.add_argument(
        "--starting_seed", type=int, default=1, help="Seed to start from (must be > 0)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="", help="Path to desired output location"
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of parallel generations"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--movis", default=False, action="store_true", help="MoVi-spheres configuration"
    )
    grp.add_argument(
        "--movia", default=False, action="store_true", help="MoVi-A configuration"
    )
    grp.add_argument(
        "--movib", default=False, action="store_true", help="MoVi-B configuration"
    )
    ret = main(parser.parse_args())
    sys.exit(ret)
