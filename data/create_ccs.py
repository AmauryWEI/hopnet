# File:         create_ccs.py
# Date:         2024/11/12
# Description:  Top-level script to create Combinatorial Complexes for different models

import argparse
from multiprocessing import Pool
import json
import gzip
from multiprocessing import cpu_count
import os
from pathlib import Path
from sys import exit, path as spath

# Add the parent directory to the Python path
spath.append(os.path.join(Path(__file__).parent.resolve(), ".."))

import numpy as np
from toponetx.classes import CombinatorialComplex
import torch
from tqdm import tqdm
import trimesh

from utils.collisions import compute_collisions_timeseries
from utils.complexes import (
    compute_nodes_and_objects_positions,
    compute_static_neighborhoods,
    compute_static_neighborhoods_extra,
    compute_dynamic_neighborhoods,
    compute_dynamic_neighborhoods_extra,
)
from utils.features import (
    compute_collision_features,
    get_feature_matrices,
    get_target_matrices,
    get_nodes_learning_masks,
)

# Environment Configuration
SCRIPT_DIR: str = os.path.dirname(os.path.realpath(__file__))
MESHES_LOCATION: str = os.path.join(SCRIPT_DIR, "objects")
METADATA_FILENAME: str = "metadata.json"
COLLISIONS_FILENAME: str = "collisions.json"
MESH_FILENAME: str = "collision_geometry.obj"

# Dataset Configuration
H: int = 2  # Past-horizon for node features (must be > 0)
OUTPUT_FILENAME: str = "complexes"
COMPRESSED_EXTENSION: str = ".npy.gz"

# Constants (do not modify)
CUBE_MESH_PATH: str = os.path.join(MESHES_LOCATION, "cube", MESH_FILENAME)
CYLINDER_MESH_PATH: str = os.path.join(MESHES_LOCATION, "cylinder", MESH_FILENAME)
SPHERE_MESH_PATH: str = os.path.join(MESHES_LOCATION, "sphere", MESH_FILENAME)
FLOOR_FRICTION: float = 0.30  # Constant from Kubric
FLOOR_RESTITUTION: float = 0.50  # Constant from Kubric
FLOOR_SIZE: float = 20.0  # Minimal approximation, Kubric uses 40
FLOOR = trimesh.Trimesh(  # Basic custom floor with 2 triangles
    vertices=[
        [-FLOOR_SIZE, -FLOOR_SIZE, 0],
        [-FLOOR_SIZE, FLOOR_SIZE, 0],
        [FLOOR_SIZE, -FLOOR_SIZE, 0],
        [FLOOR_SIZE, FLOOR_SIZE, 0],
    ],
    faces=[[0, 1, 2], [1, 2, 3]],
)
VIRTUAL_NODE_STARTING_ID: int = 9900


def get_mesh(shape: str) -> trimesh.Trimesh:
    """Load the mesh of an object bsed on its name

    Parameters
    ----------
    shape : str
        Name of the object

    Returns
    -------
    trimesh.Trimesh
        Object mesh
    """
    mesh_path: str = os.path.join(MESHES_LOCATION, shape, MESH_FILENAME)

    if os.path.exists(mesh_path):
        return trimesh.load(mesh_path)
    else:
        raise RuntimeError(f"Unknown shape {shape}")


def create_timestamped_ccc(
    t: int,
    collisions: list[dict],
    base: CombinatorialComplex,
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    nodes_positions: np.ndarray,
    nodes_target_a: np.ndarray,
    nodes_feature_v: np.ndarray,
    objects_positions: np.ndarray,
    objects_target_a: np.ndarray,
    objects_feature_v: np.ndarray,
    triangles_ids: list[np.ndarray],
    nox4: bool,
    min_z_collision_velocity: float,
) -> tuple[CombinatorialComplex, np.ndarray]:
    """Create a spatio-temporal combinatorial complex to represent the scene at time t

    Parameters
    ----------
    t : int
        Target timestep
    collisions : list[dict]
        List of collisions occuring between mesh faces
    base : CombinatorialComplex
        Common combinatorial complex containing all [0, 1, 4]-rank cells without features
    objects : list[dict]
        Objects metadata (ordered)
    meshes : list[trimesh.Trimesh]
        Objects meshes (ordered)
    nodes_positions : np.ndarray
        Nodes 3D positions at all timesteps (timestep, node_idx, 3)
    nodes_target_a : np.ndarray
        Nodes learning target 3D acceleration at all timesteps (timestep, node_idx, 3)
    nodes_feature_v : np.ndarray
        Nodes learning feature 3D velocity at all timesteps (timestep, node_idx, 3)
    objects_positions : np.ndarray
        Objects 3D positions at all timesteps (timestep, node_idx, 3)
    objects_target_a : np.ndarray
        Objects learning target 3D acceleration at all timesteps (timestep, node_idx, 3)
    objects_feature_v : np.ndarray
        Objects learning feature 3D velocity at all timesteps (timestep, node_idx, 3)
    triangles_ids : list[np.ndarray]
        Nodes involved in each triangle
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet
    min_z_collision_velocity : float
        Minimum collision velocity in the Z-axis to keep (below will be masked)

    Returns
    -------
    CombinatorialComplex
        Combinatorial complex representing timestep t (with all cells and features)
    """
    # Basic Input Validation
    assert nodes_positions.shape[0] == objects_positions.shape[0]
    assert nodes_positions.shape[2] == 3
    assert nodes_positions.shape[2] == objects_positions.shape[2]
    assert objects_positions.shape[1] == len(meshes)
    assert len(objects) == len(meshes)
    timesteps: int = nodes_positions.shape[0]
    assert t >= H and t < timesteps - 1

    # Create a copy of the base Combinatorial Complex
    complex = base.clone()
    cells_attr = {}

    # Copy of the object features (for easy assignment)
    all_obj_features: list = []

    # For each object, compute its features
    nodes_count: int = 0
    obj_idx_from_node_idx: dict = {}  # Mapping between node idx and its object idx
    for i, (object, mesh) in enumerate(zip(objects, meshes)):
        cell = np.arange(nodes_count, nodes_count + mesh.vertices.shape[0])
        for node_idx in range(nodes_count, nodes_count + mesh.vertices.shape[0]):
            obj_idx_from_node_idx[node_idx] = i
        nodes_count += mesh.vertices.shape[0]

        # Determine the type of object (static [0, 1] or moving [1, 0])
        obj_type = [0.0, 1.0] if object["asset_id"] == "floor" else [1.0, 0.0]

        # Create the learnable features based on the predefined horizon
        features = []
        for h in range(H):
            features.append(objects_feature_v[t - h, i, :])
            features.append(np.linalg.norm(features[-1]))
        features.append(np.array(obj_type))
        features.append(
            np.array([object["mass"], object["friction"], object["restitution"]])
        )
        features = np.hstack(features)

        all_obj_features.append(
            np.hstack(
                [obj_type, object["mass"], object["friction"], object["restitution"]]
            )
        )

        # Set the object's attributes to its matching 4-rank cell
        object_attr = {
            tuple(cell.tolist()): {
                "type": obj_type,
                "mass": object["mass"],
                "friction": object["friction"],
                "restitution": object["restitution"],
                "size": object["size"],
                "obj_position": objects_positions[t, i, :],
                "obj_target_acc": objects_target_a[t, i, :],
                "obj_features": features,
            }
        }

        # For the ablation study, add a virtual node at the center of each object
        if nox4:
            virtual_node_attr = {
                (VIRTUAL_NODE_STARTING_ID + i,): {
                    "position": objects_positions[t, i, :],
                    "target_acc": objects_target_a[t, i, :],
                    "features": np.hstack(
                        # Horizon velocities + center-mass distance + object features
                        [features[:8], np.zeros(8), all_obj_features[i]]
                    ),
                }
            }
            cells_attr.update(virtual_node_attr)
        else:
            cells_attr.update(object_attr)

    # For each node, compute its features
    for idx in range(nodes_positions.shape[1]):
        obj_idx: int = obj_idx_from_node_idx[idx]

        # Create the learnable features based on the predefined horizon
        features = []
        for h in range(H):
            features.append(nodes_feature_v[t - h, idx, :])
            features.append(np.linalg.norm(features[-1]))

        # Distance between node and object center (reference and now)
        features.append(nodes_positions[0, idx, :] - objects_positions[0, obj_idx, :])
        features.append(np.linalg.norm(features[-1]))
        features.append(nodes_positions[t, idx, :] - objects_positions[t, obj_idx, :])
        features.append(np.linalg.norm(features[-1]))

        # Append object-level features for ablation study (without object cells)
        if nox4:
            features.append(all_obj_features[obj_idx])

        features = np.hstack(features)

        node_attr = {
            (idx,): {
                "position": nodes_positions[t, idx, :],
                "target_acc": nodes_target_a[t, idx, :],
                "features": features,
            }
        }
        cells_attr.update(node_attr)

    # For each edge, compute its features
    edges = [list(c) for c in base.cells if len(c) == 2]
    for e in edges:
        s, d = e  # (source, destination)
        # Handle special case for ablation study (virtual nodes at objects center)
        if s >= VIRTUAL_NODE_STARTING_ID:
            s_og_position = objects_positions[0, s - VIRTUAL_NODE_STARTING_ID, :]
            s_current_position = objects_positions[t, s - VIRTUAL_NODE_STARTING_ID, :]
        else:
            s_og_position = nodes_positions[0, s, :]
            s_current_position = nodes_positions[t, s, :]
        if d >= VIRTUAL_NODE_STARTING_ID:
            d_og_position = objects_positions[0, d - VIRTUAL_NODE_STARTING_ID, :]
            d_current_position = objects_positions[t, d - VIRTUAL_NODE_STARTING_ID, :]
        else:
            d_og_position = nodes_positions[0, d, :]
            d_current_position = nodes_positions[t, d, :]
        # OG distance: distance in the original mesh position
        og_distance = s_og_position - d_og_position
        og_norm = np.linalg.norm(og_distance)
        # Distance: Distance in the current mesh position
        distance = s_current_position - d_current_position
        norm = np.linalg.norm(distance)
        # Concatenate both distances (in reference mesh and right now) like in FIGNet
        features = np.hstack([og_distance, og_norm, distance, norm])
        cells_attr.update({(s, d): {"edge_features": features, "distance": distance}})

    obj_learning_mask = np.ones((len(objects),), dtype=float)
    # Add 2-rank cells and 3-rank cells between colliding faces
    for collision in collisions:
        discarded_collisions: int = 0
        total_collisions: int = 0

        # Get IDs of the 2 colliding objects
        obj0_id = collision["obj0"]
        obj1_id = collision["obj1"]
        assert obj0_id != obj1_id

        # Process each face-to-face pair
        for pair in collision["pairs"]:
            # Get the node IDs of the face for obj0
            obj0_face = pair[0]
            obj0_node_ids = tuple(triangles_ids[obj0_id][obj0_face])
            # Get the node IDs of the face for obj1
            obj1_face = pair[1]
            obj1_node_ids = tuple(triangles_ids[obj1_id][obj1_face])
            # Sanity check
            assert obj0_node_ids != obj1_node_ids

            # Mask the learning of the objects if they are too slow in Z-axis
            if (
                np.abs(objects_feature_v[t, obj0_id, 2])
                < min_z_collision_velocity / 240
            ) and obj_learning_mask[obj0_id] > 0:
                print(
                    f"t={t}| Masking obj{obj0_id} z_vel={objects_feature_v[t, obj0_id, 2]}"
                )
                obj_learning_mask[obj0_id] = 0.0
            if (
                np.abs(objects_feature_v[t, obj1_id, 2])
                < min_z_collision_velocity / 240
            ) and obj_learning_mask[obj1_id] > 0:
                print(
                    f"t={t}| Masking obj{obj1_id} z_vel={objects_feature_v[t, obj1_id, 2]}"
                )
                obj_learning_mask[obj1_id] = 0.0

            # Compute the properties of the 3-rank cell
            safe, col_features, t0_n, t1_n = compute_collision_features(
                nodes_positions[t, obj0_node_ids, :],
                nodes_positions[t, obj1_node_ids, :],
            )

            total_collisions += 1
            # If the collision is ill-defined (triangles intersecting), discard it
            if not safe:
                discarded_collisions += 1
                continue

            # Add a 2-rank cell for each face
            complex.add_cell(sorted(obj0_node_ids), rank=2)
            complex.add_cell(sorted(obj1_node_ids), rank=2)
            # Add a 3-rank cell as colliding relationship
            complex.add_cell(sorted(obj0_node_ids) + sorted(obj1_node_ids), rank=3)

            cells_attr.update(
                {
                    tuple(sorted(obj0_node_ids)): {"face_features": t0_n},
                    tuple(sorted(obj1_node_ids)): {"face_features": t1_n},
                    tuple(sorted(obj0_node_ids) + sorted(obj1_node_ids)): {
                        "col_features": col_features,
                        # Track the order of faces to sign the incidence matrix
                        "face_0": tuple(sorted(obj0_node_ids)),
                    },
                }
            )

    # WARNING: CELL ATTRIBUTES MUST BE SET LAST (after creating collisions)
    complex.set_cell_attributes(cells_attr)

    return complex, obj_learning_mask


def compute_neighborhoods_multi(cccs: list[CombinatorialComplex]) -> tuple:
    """Compute neighborhood matrices required by default HOPNet model

    Parameters
    ----------
    cccs : list[CombinatorialComplex]
        Spatio-temporal combinatorial complexes

    Returns
    -------
    tuple
        Adjacency and incidence matrices as sparse COO Pytorch tensors
    """
    b02_l = []
    b04 = compute_static_neighborhoods(cccs[0])[0]
    b12_l = []
    b23_l = []
    b24_l = []

    for ccc in cccs:
        n = compute_dynamic_neighborhoods(ccc)
        b02_l.append(n[0])
        b12_l.append(n[1])
        b23_l.append(n[2])
        b24_l.append(n[3])

    return (b02_l, b04, b12_l, b23_l, b24_l)


def compute_neighborhoods_multi_extra(cccs: list[CombinatorialComplex]) -> tuple:
    """Compute additional neighborhoods (not required by default HOPNet model)

    Parameters
    ----------
    cccs : list[CombinatorialComplex]
        Spatio-temporal combinatorial complexes

    Returns
    -------
    tuple
        Adjacency and incidence matrices as sparse COO Pytorch tensors
    """
    a010, a101, b01, b14 = compute_static_neighborhoods_extra(cccs[0])
    a232_l = []
    b03_l = []
    b13_l = []

    for ccc in cccs:
        n = compute_dynamic_neighborhoods_extra(ccc)
        a232_l.append(n[0])
        b03_l.append(n[1])
        b13_l.append(n[2])

    return (a010, a101, a232_l, b01, b03_l, b13_l, b14)


def create_ccs_for_sample(
    sample_dir: str,
    output_filename: str,
    collision_radius: float,
    nox4: bool,
    min_z_collision_velocity: float,
    processes: int,
    save: bool = True,
) -> dict:
    """Create Combinatorial Complexes for a given sample in a dataset

    Parameters
    ----------
    sample_dir : str
        Absolute path to the sample directory
    output_filename : str
        Name of the output file to generate
    collision_radius : float
        Threshold distance between faces to assume a collision
    nox4 : bool
        True to generate CCs for ablation study, False for standard CCs
    min_z_collision_velocity : float
        Minimum collision velocity in the Z-axis to keep (below will be masked)
    processes : int
        Maximum number of parallel processes to spawn
    save : bool, optional
        Export the computed CCs to the disk, by default True

    Returns
    -------
    dict
        Computed Combinatorial Complexes and neighborhood matrices
    """
    # Load the metadata
    metadata_path: str = os.path.join(sample_dir, METADATA_FILENAME)
    with open(metadata_path) as f:
        metadata: dict = json.load(f)

    # Minimal Input Validation
    num_objects: int = metadata["metadata"]["num_instances"]
    objects: list[dict] = metadata["instances"]
    timesteps: int = len(objects[0]["positions"])
    assert num_objects == len(objects)

    # Load the base meshes for each object in the experiment
    meshes: list[trimesh.Trimesh] = []
    for object in objects:
        scale: float = object["size"]
        mesh: trimesh.Trimesh = get_mesh(object["shape"])
        mesh.apply_scale(scale)
        meshes.append(mesh)

    # Add floor data
    floor_metadata = {
        "asset_id": "floor",
        "angular_velocities": np.zeros((timesteps, 3)).tolist(),
        "friction": FLOOR_FRICTION,
        "mass": 0.0,
        "positions": np.zeros((timesteps, 3)).tolist(),
        "quaternions": np.repeat([[1.0, 0.0, 0.0, 0.0]], timesteps, axis=0).tolist(),
        "restitution": FLOOR_RESTITUTION,
        "size": 1.0,
        "velocities": np.zeros((timesteps, 3)).tolist(),
    }

    objects.append(floor_metadata)
    meshes.append(FLOOR)

    # Check if pre-computed collisions are available
    if os.path.exists(os.path.join(sample_dir, COLLISIONS_FILENAME)):
        with open(os.path.join(sample_dir, COLLISIONS_FILENAME), "r") as f:
            collisions = json.load(f)
    else:
        # Compute the colliding faces for the whole experiment
        assert collision_radius > 0
        collisions = compute_collisions_timeseries(objects, meshes, collision_radius)
        with open(os.path.join(sample_dir, COLLISIONS_FILENAME), "w") as f:
            json.dump(collisions, f)

    # Compute the position of each node and each object
    (
        nodes_positions,  # [all_timesteps, nodes, 3]
        nodes_target_a,  # [all_timesteps, nodes, 3] (first and last timesteps are NaN)
        nodes_feature_v,  # [all_timesteps, nodes, 3] (first timestep is NaN)
        objects_positions,
        objects_target_a,
        objects_feature_v,
        base_ccc,
        triangles_ids,
    ) = compute_nodes_and_objects_positions(objects, meshes, nox4)

    cccs: list[CombinatorialComplex] = []
    obj_learning_masks: list[np.ndarray] = []
    # For each timestep, create a new CCC and apply the positions
    if processes > 1:
        # Multiprocessing if requested
        pool = Pool(processes=processes)
        results = [
            pool.apply_async(
                create_timestamped_ccc,
                args=(
                    t,
                    collisions[str(t)],
                    base_ccc,
                    objects,
                    meshes,
                    nodes_positions,
                    nodes_target_a,
                    nodes_feature_v,
                    objects_positions,
                    objects_target_a,
                    objects_feature_v,
                    triangles_ids,
                    nox4,
                    min_z_collision_velocity,
                ),
            )
            for t in range(H, timesteps - 1)
        ]
        results = [r.get() for r in results]
        cccs = [r[0] for r in results]
        obj_learning_masks = [r[1] for r in results]
    else:
        # If no multiprocessing, compute everything here step by step
        for t in range(H, timesteps - 1):
            ccc, obj_learning_mask = create_timestamped_ccc(
                t,
                collisions[str(t)],
                base_ccc,
                objects,
                meshes,
                nodes_positions,
                nodes_target_a,
                nodes_feature_v,
                objects_positions,
                objects_target_a,
                objects_feature_v,
                triangles_ids,
                nox4,
                min_z_collision_velocity,
            )
            cccs.append(ccc)
            obj_learning_masks.append(obj_learning_mask)

    # For each CCC, compute its adjacency and incidence matrices
    b02_l, b04, b12_l, b23_l, b24_l = compute_neighborhoods_multi(cccs)
    a010, a101, a232_l, b01, b03_l, b13_l, b14 = compute_neighborhoods_multi_extra(cccs)

    x0, x1, x2_l, x3_l, x4 = get_feature_matrices(cccs, horizon=H)
    t0, t4 = get_target_matrices(cccs)
    x4_mask = np.stack(obj_learning_masks)
    x0_mask = get_nodes_learning_masks(x4_mask, b04)

    data = {
        "x0": x0,
        "x1": x1,
        "x2_l": x2_l,
        "x3_l": x3_l,
        "x4": x4,
        "t0": t0,
        "t4": t4,
        "b02_l": b02_l,
        "b04": b04,
        "b12_l": b12_l,
        "b23_l": b23_l,
        "b24_l": b24_l,
        "a010": a010,
        "a101": a101,
        "a232_l": a232_l,
        "b01": b01,
        "b03_l": b03_l,
        "b13_l": b13_l,
        "b14": b14,
        "x0_mask": x0_mask.float(),
        "x4_mask": torch.from_numpy(x4_mask).float(),
        "seed": metadata["metadata"]["seed"],
    }

    # Save a dictionary with the complexes to a compressed Numpy `.npy.gz` file
    if save:
        f = gzip.GzipFile(
            os.path.join(sample_dir, output_filename + COMPRESSED_EXTENSION), "w"
        )
        np.save(f, data)  # type: ignore
        f.close()

    return data


def main(args: argparse.Namespace) -> int:
    # Minimal argument checking
    assert args.collision_radius > 0

    # Make sure the dataset exists
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset dir {dataset_dir} does not exist.")

    # List the samples in the dataset
    samples = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
    try:
        seeds = set([int(f) for f in samples])
        print(f"INFO: Found {len(seeds)} samples in dataset directory.")
    except Exception as e:
        print(f"ERROR: Dataset directory contains non-digit folders.")
        print("\t=>", e)
        return 1

    output_filename = OUTPUT_FILENAME + ("-nox4" if args.nox4 else "")

    samples_full = [os.path.join(dataset_dir, exp) for exp in samples]
    already_computed: list[str] = []
    empty_samples: list[str] = []
    for exp in samples_full:
        # Check if the dataset samples have already been processed
        out_f = os.path.join(exp, output_filename + COMPRESSED_EXTENSION)
        if os.path.isfile(out_f) and os.stat(out_f).st_size > 0:
            already_computed.append(exp)

        # Check if directories are empty
        meta_f = os.path.join(exp, METADATA_FILENAME)
        if not (os.path.isfile(meta_f) and os.stat(meta_f).st_size > 0):
            empty_samples.append(exp)

    [samples_full.remove(a) for a in already_computed]
    [samples_full.remove(a) for a in empty_samples]
    print(f"INFO: Found {len(already_computed)} already pre-processed samples.")
    print(f"WARN: Found {len(empty_samples)} empty samples: {empty_samples}")

    # Among the remaining samples, select the ones in the requested range
    samples_ids = sorted([int(os.path.basename(e)) for e in samples_full])
    if args.samples == 0:  # If requested to process all samples
        target_seeds = sorted([id for id in samples_ids if id >= args.starting_seed])
    else:
        target_seeds = sorted(
            [
                id
                for id in samples_ids
                if (
                    id >= args.starting_seed
                    and id < args.starting_seed + args.experiments
                )
            ]
        )
    target_samples = [os.path.join(dataset_dir, str(e)) for e in target_seeds]
    print(f"INFO: Computing CCs for {len(target_seeds)} experiments")

    # For each experiment, create a set of Combinatorial Complex (1 per timestep)
    for exp in tqdm(target_samples):
        create_ccs_for_sample(
            exp,
            output_filename,
            args.collision_radius,
            args.nox4,
            args.min_z_collision_velocity,
            processes=args.threads,
            save=True,
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CCs to train HOPNet models")
    parser.add_argument("dataset_dir", type=str, help="Location of the MoVi dataset")
    parser.add_argument(
        "--nox4",
        action="store_true",
        default=False,
        help="For ablation study without object cells X4",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=cpu_count(),
        help="Max number of CPU threads to use (for parallelization)",
    )
    parser.add_argument(
        "--collision_radius",
        type=float,
        default=0.1,
        help="Collision detection radius (must be > 0)",
    )
    parser.add_argument(
        "--starting_seed", type=int, default=1, help="Seed to start from (must be > 0)"
    )
    parser.add_argument(
        "--samples", type=int, default=0, help="Number of samples to process (0 = all)"
    )
    parser.add_argument(
        "--min_z_collision_velocity",
        type=float,
        default=0.0,
        help="Minimum Z-axis collision velocity",
    )
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
