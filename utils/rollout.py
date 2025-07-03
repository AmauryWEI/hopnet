# File:         rollout.py
# Date:         2024/05/22
# Description:  Contains functions to run auto-regressive rollout

from copy import deepcopy
from os import path as opath
from pathlib import Path
from sys import path as spath

# Add the parent directory to the Python path
spath.append(opath.join(Path(__file__).parent.resolve(), ".."))

import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from toponetx.classes import CombinatorialComplex
import torch
import trimesh

from .collisions import compute_collisions
from .complexes import (
    compute_static_neighborhoods,
    compute_static_neighborhoods_extra,
    compute_dynamic_neighborhoods,
    compute_dynamic_neighborhoods_extra,
)
from .features import (
    compute_feature_velocity,
    compute_collision_features,
    get_feature_matrices,
)
from data.movi_dataset import (
    MoviNormalization,
    normalize,
    denormalize,
)

VIRTUAL_NODE_STARTING_ID: int = 9900


def build_base_complex(
    meshes: list[trimesh.Trimesh], objects: list[dict], nox4: bool
) -> tuple[CombinatorialComplex, list[np.ndarray], dict]:
    """Create the base Combinatorial Complex for fixed scenario

    No cell features, no 2-rank or 3-rank cells are added.

    Parameters
    ----------
    meshes : list[trimesh.Trimesh]
        Objects meshes
    objects : list[dict]
        Objects data

    Returns
    -------
    CombinatorialComplex
        Base CCC with all objects nodes as 0-rank, edges as 1-rank, and objects as 4-rank
    list[np.ndarray]
        Triangle IDs of each object's faces with the global nodes numbering
    dict
        Correspondence between node index and its respective object index
    """
    scene: trimesh.Scene = trimesh.Scene()  # Used to have all nodes in a single mesh
    nodes_count: int = 0  # Use to keep track of the number of nodes per object
    obj_idx_from_node_idx: dict = {}  # Mapping between node idx and its object idx
    triangles_ids: list[np.ndarray] = []  # (object, triangle_id, global_node_ids)

    for i, (object, og_mesh) in enumerate(zip(objects, meshes)):
        # Gather its transformation
        transformation = trimesh.transformations.quaternion_matrix(
            object["quaternions"][0]
        )
        transformation[:3, 3] = object["positions"][0]

        # Add the individual object to the global environment
        scene.add_geometry(og_mesh, transform=transformation)

        # Store the triangle IDs with the global node numbering (instead of per-object)
        triangles_ids.append(np.array(og_mesh.faces + nodes_count))

        # Store the mapping between nodes and objects IDs
        for node_idx in range(nodes_count, nodes_count + og_mesh.vertices.shape[0]):
            obj_idx_from_node_idx[node_idx] = i

        # Increase the number of global nodes added
        nodes_count += og_mesh.vertices.shape[0]

    global_mesh = scene.to_mesh()

    # Create the CCC
    complex: CombinatorialComplex = CombinatorialComplex()
    # Add all nodes (0-rank)
    [complex.add_cell(v, rank=0) for v in range(global_mesh.vertices.shape[0])]
    # Add all edges (1-rank)
    [complex.add_cell(e.tolist(), rank=1) for e in global_mesh.edges]

    if nox4:
        nodes_count: int = 0
        # For ablation study, add a virtual node at the center of the object
        for i in range(len(objects)):
            complex.add_cell(VIRTUAL_NODE_STARTING_ID + i, rank=0)
            # Add edges between all nodes of an object and its virtual node
            for j in range(meshes[i].vertices.shape[0]):
                complex.add_cell(
                    [VIRTUAL_NODE_STARTING_ID + i, nodes_count + j], rank=1
                )
            nodes_count += meshes[i].vertices.shape[0]

    # Add all objects (4-rank)
    nodes_count: int = 0
    for mesh in meshes:
        cell = np.arange(nodes_count, nodes_count + mesh.vertices.shape[0])
        complex.add_cell(cell.tolist(), rank=4)
        nodes_count += mesh.vertices.shape[0]

    return complex, triangles_ids, obj_idx_from_node_idx


def build_featured_complex(
    base_complex: CombinatorialComplex,
    collisions: list[dict] | None,
    triangle_ids: list[np.ndarray],
    obj_idx_from_node_idx: dict,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    nox4: bool,
    horizon: int = 2,
    collision_radius: float = 0.10,
) -> tuple[CombinatorialComplex, np.ndarray, float]:
    """Build a single Combinatorial Complex for one timestep to use as model input

    Parameters
    ----------
    base_complex : CombinatorialComplex
        Base CCC with the proper 0-rank, 1-rank, and 4-rank cells
    collisions : list[dict] | None
        Collisions at the current timestep
    triangle_ids : list[np.ndarray]
        Triangle IDs of each object's faces with the global nodes numbering
    obj_idx_from_node_idx : dict
        Correspondence between nodes and their respective object index
    objects: list[dict]
        List of objects, containing properties mass, friction, restituion, ...
    obj_pos : np.ndarray
        Objects' positions ; shape [horizon+1, objs_count, 3]
    obj_quat : np.ndarray
        Objects' quaternions ; shape [horizon+1, objs_count, 4] [w, x, y, z]
    meshes : list[trimesh.Trimesh]
        Original objects' meshes
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet
    horizon : int
        Past horizon H to use as node features
    collision_radius : int
        Collision radius to detect collision between mesh faces

    Returns
    -------
    CombinatorialComplex
        CCC with the right cells and features
    np.ndarray
        Nodes positions ; shape [horizon+1, nodes_count, 3]
    """
    # Minimal input validation
    assert horizon > 0
    assert collision_radius > 0
    assert len(objects) == len(meshes)
    assert obj_pos.shape[0] == horizon + 1
    assert obj_quat.shape[0] == horizon + 1
    assert obj_pos.shape[1] == len(meshes)
    assert obj_quat.shape[1] == len(meshes)
    assert obj_pos.shape[2] == 3
    assert obj_quat.shape[2] == 4

    ccc = base_complex.clone()
    cells_attr: dict = {}

    # Step 0: Compute the collisions between objects
    if collisions is None:
        collisions = compute_collisions(
            obj_pos[-1], obj_quat[-1], meshes, collision_radius
        )

    # Step 1: Compute the objects' features
    objs_attr, objs_features = _compute_objs_attrs(
        objects, meshes, obj_pos, horizon, nox4
    )
    cells_attr.update(objs_attr)

    # Step 2: Compute the nodes' features
    nodes_edges_attr, nodes_pos = _compute_nodes_edges_attrs(
        objects,
        meshes,
        obj_pos,
        obj_quat,
        [list(e) for e in base_complex.cells if len(e) == 2],  # type: ignore
        horizon,
        nox4,
        obj_idx_from_node_idx,
        objs_features,
    )
    cells_attr.update(nodes_edges_attr)

    # Step 3: Compute the collisions' features
    ccc, collisions_attr, invalid_collisions_ratio = _add_collisions_attrs(
        ccc, collisions, nodes_pos[-1], objs_features, triangle_ids
    )
    cells_attr.update(collisions_attr)

    # WARNING: CELL ATTRIBUTES MUST BE SET LAST (after creating collisions)
    ccc.set_cell_attributes(cells_attr)

    # Short validation
    if nox4:
        assert len(ccc.get_cell_attributes("features")) == nodes_pos.shape[1] + len(
            objects
        )
    else:
        assert len(ccc.get_cell_attributes("features")) == nodes_pos.shape[1]
        assert len(ccc.get_cell_attributes("obj_features")) == len(objects)

    return ccc, nodes_pos, invalid_collisions_ratio


def model_input_from_ccc(
    ccc: CombinatorialComplex,
    horizon: int,
    norm: MoviNormalization | None,
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
) -> tuple:
    """Create a tuple to use as direct model input

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combinatorial Complex with the proper cells and cells features
    horizon : int
        Horizon (used to pre-allocate the matrices)
    norm : MoviNormalization
        Normalization structure

    Returns
    -------
    tuple
        Ordered inputs to pass to a TDL model
    """
    x0, x1, x2, x3, x4 = get_feature_matrices([ccc], horizon)
    t0_zero_acc = torch.zeros(3)
    t4_zero_acc = torch.zeros(3)

    if norm:
        x0 = normalize(x0[0], norm.x0_mean, norm.x0_std)
        x1 = normalize(x1[0], norm.x1_mean, norm.x1_std)
        x2 = normalize(x2[0], norm.x2_mean, norm.x2_std) if x2[0] is not None else None
        x3 = normalize(x3[0], norm.x3_mean, norm.x3_std) if x3[0] is not None else None
        t0_zero_acc = normalize(t0_zero_acc, norm.t0_mean, norm.t0_std)
        if model_name != "NoObjectCells":
            x4 = normalize(x4[0], norm.x4_mean, norm.x4_std)
            t4_zero_acc = normalize(t4_zero_acc, norm.t4_mean, norm.t4_std)
    else:
        x0 = x0[0]
        x1 = x1[0]
        x2 = x2[0] if x2[0] is not None else None
        x3 = x3[0] if x3[0] is not None else None
        x4[0] = x4[0]

    b04 = compute_static_neighborhoods(ccc, device)[0]
    (b02, b12, b23, b24) = compute_dynamic_neighborhoods(ccc, device)
    if model_name in ["NoSequential", "NoObjectCells"]:
        a010, a101, b01, b14 = compute_static_neighborhoods_extra(ccc, device)
        a232, b03, b13 = compute_dynamic_neighborhoods_extra(ccc, device)

    if model_name == "NoSequential":
        m20 = torch.zeros((x0.shape[0], model.hid_channels[0]), device=device)
        m21 = torch.zeros((x1.shape[0], model.hid_channels[1]), device=device)
        m24 = torch.zeros((x4.shape[0], model.hid_channels[4]), device=device)
        return (
            x0.to(device),
            x1.to(device),
            x2.to(device) if x2 is not None else None,
            x3.to(device) if x3 is not None else None,
            x4.to(device),
            t0_zero_acc.to(device),
            t4_zero_acc.to(device),
            a010.to(device),
            a101.to(device),
            a232.to(device) if a232 is not None else None,
            b01.to(device),
            b02.to(device) if b02 is not None else None,
            b03.to(device) if b03 is not None else None,
            b04.to(device),
            b12.to(device) if b12 is not None else None,
            b13.to(device) if b13 is not None else None,
            b14.to(device) if b14 is not None else None,
            b23.to(device) if b23 is not None else None,
            b24.to(device) if b24 is not None else None,
            m20,
            m21,
            m24,
        )
    elif model_name == "NoObjectCells":
        return (
            x0.to(device),
            x1.to(device),
            x2.to(device) if x2 is not None else None,
            x3.to(device) if x3 is not None else None,
            t0_zero_acc.to(device),
            a010.to(device),
            b02.to(device) if b02 is not None else None,
            b12.to(device) if b12 is not None else None,
            b23.to(device) if b23 is not None else None,
        )
    else:
        return (
            x0.to(device),
            x1.to(device),
            x2.to(device) if x2 is not None else None,
            x3.to(device) if x3 is not None else None,
            x4.to(device),
            t0_zero_acc.to(device),
            t4_zero_acc.to(device),
            b02.to(device) if b02 is not None else None,
            b04.to(device),
            b12.to(device) if b12 is not None else None,
            b23.to(device) if b23 is not None else None,
            b24.to(device) if b24 is not None else None,
        )


def positions_from_model_output(
    out0: torch.Tensor,
    norm: MoviNormalization | None,
    nodes_pos: np.ndarray,
) -> np.ndarray:
    """Compute nodes and objects' positions from a TDL model output

    Parameters
    ----------
    out0 : torch.Tensor
        Output nodes' embeddings of the model
    norm : MoviNormalization | None
        Normalization (optional)
    nodes_pos : np.ndarray
        Previous nodes' positions (2 previous timesteps)

    Returns
    -------
    np.ndarray
        Predicted nodes positions
    """
    # Minimal input validation
    assert out0.shape[1] == 3
    assert nodes_pos.shape[0] == 2  # Need the current timestep and the previous
    assert nodes_pos.shape[1] == out0.shape[0]
    assert nodes_pos.shape[2] == 3

    nodes_acc = out0.cpu()

    if norm:
        pred_nodes_acc = denormalize(nodes_acc, norm.t0_mean, norm.t0_std)

    pred_nodes_pos = pred_nodes_acc + 2 * nodes_pos[-1] - nodes_pos[-2]

    return pred_nodes_pos


def shape_matching(
    nodes_pos: np.ndarray, meshes: list[trimesh.Trimesh]
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate multi objects' positions and orientations with shape matching

    Parameters
    ----------
    nodes_pos : np.ndarray
        Nodes positions of all objects (in order)
    meshes : list[trimesh.Trimesh]
        Meshes

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Objects' estimated positions, Object's estimated quaternions [w, x, y, z]
    """
    # Minimal input validation
    assert len(meshes) > 0
    assert nodes_pos.shape[1] == 3

    # Get the number of nodes per object
    nodes_per_obj: list[int] = [0]
    for mesh in meshes:
        nodes_per_obj.append(nodes_per_obj[-1] + mesh.vertices.shape[0])
    assert nodes_per_obj[-1] == nodes_pos.shape[0]

    objs_pos = np.zeros((len(meshes), 3))
    objs_quat = np.zeros((len(meshes), 4))

    # For each object, estimate its rotation and translation
    for i in range(0, len(meshes)):
        og_center_mass = meshes[i].center_mass
        # Create a copy of the rotated mesh (to get the center of mass)
        rot_mesh = deepcopy(meshes[i])
        rot_mesh.vertices = nodes_pos[nodes_per_obj[i] : nodes_per_obj[i + 1]]
        rot_center_mass = rot_mesh.center_mass

        qi = meshes[i].vertices - og_center_mass
        pi = rot_mesh.vertices - rot_center_mass
        est_rot = _shape_match(qi, pi)

        # Check if the position is actually correct here
        objs_pos[i] = rot_center_mass - est_rot.as_matrix() @ og_center_mass
        objs_quat[i] = est_rot.as_quat(False)[[-1, 0, 1, 2]]  # [w, x, y, z]

    return objs_pos, objs_quat


def _compute_objs_attrs(
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    obj_pos: np.ndarray,
    horizon: int,
    nox4: bool,
) -> tuple[dict, list[np.ndarray]]:
    """Compute objects (4-rank cells) attributes for a CCC

    Parameters
    ----------
    objects : list[dict]
        Objects properties (containing friction, mass, restitution, ...)
    meshes : list[trimesh.Trimesh]
        Original object's meshes
    obj_pos : np.ndarray
        Objects's positions ; shape [horizon + 1, objs_count, 3]
    horizon : int
        Past horizon H to use
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet

    Returns
    -------
    dict
        4-rank cells features
    """
    # Minimal input validation
    assert len(objects) == len(meshes)
    assert obj_pos.shape[0] == horizon + 1
    assert obj_pos.shape[1] == len(objects)
    assert obj_pos.shape[2] == 3

    cells_attr: dict = {}
    objs_features: list[np.ndarray] = []

    # For each object, compute its feature velocity
    objs_feature_v: list[np.ndarray] = []
    for obj_idx in range(obj_pos.shape[1]):
        objs_feature_v.append(compute_feature_velocity(obj_pos[:, obj_idx, :]))

    # For each object, compute its features
    nodes_count: int = 0
    for i, (object, mesh) in enumerate(zip(objects, meshes)):
        # Build the name of the 4-rank cell for the object
        cell = np.arange(nodes_count, nodes_count + mesh.vertices.shape[0])
        nodes_count += mesh.vertices.shape[0]

        # Determine the type of object (static [0, 1] or moving [1, 0])
        obj_type = [0.0, 1.0] if object["asset_id"] == "floor" else [1.0, 0.0]

        # Create the learnable features based on the predefined horizon
        features = []
        for h in range(horizon):
            features.append(objs_feature_v[i][-1 - h, :])
            features.append(np.linalg.norm(objs_feature_v[i][-1 - h, :]))
        features.append(np.array(obj_type))
        features.append(
            np.array([object["mass"], object["friction"], object["restitution"]])
        )
        features = np.hstack(features)

        # Track the objects' features, useful later for collisions
        objs_features.append(
            np.hstack(
                [obj_type, object["mass"], object["friction"], object["restitution"]]
            )
        )

        # Set the object's attributes to its matching 4-rank cell
        object_attr = {tuple(cell.tolist()): {"obj_features": features}}
        if nox4:
            # For the ablation study, add a virtual node at the center of each object
            virtual_node_attr = {
                (VIRTUAL_NODE_STARTING_ID + i,): {
                    "features": np.hstack(
                        # Horizon velocities + center-mass distance + object features
                        [features[:8], np.zeros(8), objs_features[i]]
                    ),
                }
            }
            cells_attr.update(virtual_node_attr)
        else:
            cells_attr.update(object_attr)

    return cells_attr, objs_features


def _compute_nodes_edges_attrs(
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    edges: list[tuple],
    horizon: int,
    nox4: bool,
    obj_idx_from_node_idx: dict,
    all_obj_features: list[np.ndarray],
) -> tuple[dict, np.ndarray]:
    """Compute nodes (0-rank) and edges (1-rank) attributes for a CCC

    Parameters
    ----------
    objects : list[dict]
        Objects properties (containing friction, mass, restitution, ...)
    meshes : list[trimesh.Trimesh]
        Original object's meshes
    obj_pos : np.ndarray
        Objects' positions ; shape [horizon + 1, objs_count, 3]
    obj_quat : np.ndarray
        Objects' quaternions ; shape [horizon + 1, objs_count, 4] [w, x, y, z]
    edges : list[tuple]
        Edges in the scene mesh (with global nodes numbering)
    horizon : int
        Past horizon H to use
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet
    obj_idx_from_node_idx : dict
        Correspondence between nodes and their respective object index

    Returns
    -------
    dict
        0-rank and 1-rank cells features
    np.ndarray
        Nodes positions ; shape [horizon + 1, nodes_count, 3]
    """
    # Minimal input validation
    assert len(objects) == len(meshes)
    assert obj_pos.shape[0] == horizon + 1
    assert obj_quat.shape[0] == horizon + 1
    assert obj_pos.shape[1] == len(objects)
    assert obj_quat.shape[1] == len(objects)
    assert obj_pos.shape[2] == 3
    assert obj_quat.shape[2] == 4
    assert len(edges) > 0

    cells_attr: dict = {}

    # For the given horizon, compute the positions of all nodes
    nodes_pos_list: list[np.ndarray] = []
    for t in range(horizon + 1):
        # Store the node positions
        nodes_pos_list.append(
            compute_nodes_positions(meshes, obj_pos[t, :, :], obj_quat[t, :, :])
        )
    # Shape (horizon+1, nodes, 3)
    nodes_pos = np.stack(nodes_pos_list, axis=0)
    nodes_feature_v: np.ndarray = np.zeros((horizon + 1, nodes_pos.shape[1], 3))

    # Original nodes positions (for edge features)
    og_obj_pos = np.array([obj["positions"][0] for obj in objects])
    og_obj_quat = np.array([obj["quaternions"][0] for obj in objects])
    og_nodes_pos = compute_nodes_positions(meshes, og_obj_pos, og_obj_quat)

    # For each node, compute its learnable features
    for node_idx in range(nodes_pos.shape[1]):
        obj_idx: int = obj_idx_from_node_idx[node_idx]
        # Compute the learning features
        feature_v = compute_feature_velocity(nodes_pos[:, node_idx, :])
        nodes_feature_v[:, node_idx, :] = feature_v

        # Create the learnable features based on the predefined horizon
        features = []
        for h in range(horizon):
            features.append(nodes_feature_v[-1 - h, node_idx, :])
            features.append(np.linalg.norm(nodes_feature_v[-1 - h, node_idx, :]))

        # 2nd features: distance between node and object center (reference and now)
        features.append(og_nodes_pos[node_idx, :] - og_obj_pos[obj_idx, :])
        features.append(np.linalg.norm(features[-1]))
        features.append(nodes_pos[t, node_idx, :] - obj_pos[t, obj_idx, :])
        features.append(np.linalg.norm(features[-1]))
        if nox4:
            features.append(all_obj_features[obj_idx])
        features = np.hstack(features)

        node_attr = {(node_idx,): {"features": features}}
        cells_attr.update(node_attr)

    # For each edge, compute its features
    for e in edges:
        s, d = e  # (source, destination)
        # Handle special case for ablation study (virtual nodes at objects center)
        if s >= VIRTUAL_NODE_STARTING_ID:
            s_og_position = og_obj_pos[s - VIRTUAL_NODE_STARTING_ID, :]
            s_current_position = obj_pos[-1, s - VIRTUAL_NODE_STARTING_ID, :]
        else:
            s_og_position = og_nodes_pos[s, :]
            s_current_position = nodes_pos[-1, s, :]
        if d >= VIRTUAL_NODE_STARTING_ID:
            d_og_position = og_obj_pos[d - VIRTUAL_NODE_STARTING_ID, :]
            d_current_position = obj_pos[-1, d - VIRTUAL_NODE_STARTING_ID, :]
        else:
            d_og_position = og_nodes_pos[d, :]
            d_current_position = nodes_pos[-1, d, :]
        # OG distance: distance in the original mesh position
        og_distance = s_og_position - d_og_position
        og_norm = np.linalg.norm(og_distance)
        # Distance: Distance in the current mesh position
        distance = s_current_position - d_current_position
        norm = np.linalg.norm(distance)
        # Concatenate both distances (in reference mesh and right now) like in FIGNet
        features = np.hstack([og_distance, og_norm, distance, norm])
        cells_attr.update({(s, d): {"edge_features": features}})

    return cells_attr, nodes_pos


def compute_nodes_positions(
    meshes: list[trimesh.Trimesh], obj_pos: np.ndarray, obj_quat: np.ndarray
) -> np.ndarray:
    """Compute the positions of all nodes in a scene with multiple objects

    Parameters
    ----------
    meshes : list[trimesh.Trimesh]
        Original meshes of the objects
    obj_pos : np.ndarray
        Objects' positions ; shape [objects, 3]
    obj_quat : np.ndarray
        Objects' quaternions ; shape [objects, 4] [w, x, y, z]

    Returns
    -------
    np.ndarray
        Individual nodes positions ; shape [nodes, 3]
    """
    scene: trimesh.Scene = trimesh.Scene()

    # For each object in the sample
    for obj_idx, og_mesh in enumerate(meshes):
        # Gather its transformation
        transformation = trimesh.transformations.quaternion_matrix(obj_quat[obj_idx, :])
        transformation[:3, 3] = obj_pos[obj_idx, :]

        # Append the new object to the scene
        scene.add_geometry(og_mesh, transform=transformation)

    global_mesh = scene.to_mesh()

    return global_mesh.vertices


def _add_collisions_attrs(
    ccc: CombinatorialComplex,
    collisions: list[dict],
    nodes_pos: np.ndarray,
    objs_features: list[np.ndarray],
    triangle_ids: list[np.ndarray],
) -> tuple[CombinatorialComplex, dict, float]:
    """Add faces (2-rank) and collision (3-rank) cells to a CCC, and build their features

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combinatorial Complex to add the relevant cells to
    collisions : list[dict]
        List of face-face collisions
    nodes_pos : np.ndarray
        Current nodes positions
    objs_features : list[np.ndarray]
        Object global features (to assign to the 2-rank cells)
    triangle_ids : list[np.ndarray]
        Triangle IDs with global numbering for each object

    Returns
    -------
    tuple[CombinatorialComplex, dict]
        Updated Combinatorial Complex, additional cell features
    """
    assert nodes_pos.shape[1] == 3
    assert len(objs_features) > 0
    assert len(triangle_ids) == len(objs_features)

    cells_attr: dict = {}

    discarded_collisions: int = 0
    total_collisions: int = 0
    # Add 2-rank cells and 3-rank cells between colliding faces
    for collision in collisions:
        # Get IDs of the 2 colliding objects
        obj0_id = collision["obj0"]
        obj1_id = collision["obj1"]
        assert obj0_id != obj1_id

        # Process each face-to-face pair
        for pair in collision["pairs"]:
            # Get the node IDs of the face for obj0
            obj0_face = pair[0]
            obj0_node_ids = tuple(triangle_ids[obj0_id][obj0_face])
            # Get the node IDs of the face for obj1
            obj1_face = pair[1]
            obj1_node_ids = tuple(triangle_ids[obj1_id][obj1_face])
            # Sanity check
            assert obj0_node_ids != obj1_node_ids

            # Compute the properties of the 3-rank cell
            safe, col_features, t0_n, t1_n = compute_collision_features(
                nodes_pos[obj0_node_ids, :], nodes_pos[obj1_node_ids, :]
            )

            total_collisions += 1
            # If the collision is ill-defined (triangles intersecting), discard it
            if not safe:
                discarded_collisions += 1
                continue

            # Add a 2-rank cell for each face
            ccc.add_cell(sorted(obj0_node_ids), rank=2)
            ccc.add_cell(sorted(obj1_node_ids), rank=2)
            # Add a 3-rank cell as colliding relationship
            ccc.add_cell(sorted(obj0_node_ids) + sorted(obj1_node_ids), rank=3)

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

    return (
        ccc,
        cells_attr,
        (discarded_collisions / total_collisions if total_collisions > 0 else 0),
    )


def _shape_match(qi: np.ndarray, pi: np.ndarray) -> Rotation:
    """Estimate the rotation of a single mesh from qi to pi

    Reference: https://graphics.stanford.edu/courses/cs468-05-fall/Papers/p471-muller.pdf

    Parameters
    ----------
    qi : np.ndarray
        Centered original mesh vertices
    pi : np.ndarray
        Centered rotated mesh vertices

    Returns
    -------
    np.ndarray
        Estimated quaternion rotation [w, x, y, z]
    """
    assert qi.shape == pi.shape
    assert qi.shape[1] == 3

    # Optimal Transformation A
    Apq = np.zeros((3, 3))
    for i in range(qi.shape[0]):
        Apq += 1 / qi.shape[0] * pi[i].reshape(3, 1) @ qi[i].reshape(1, 3)

    # Optimal Rotation R
    R, _ = polar(Apq)
    q = Rotation.from_matrix(R)

    return q
