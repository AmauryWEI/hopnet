# File:         complexes.py
# Date:         2024/05/15
# Description:  Contains functions to compute Combinatorial Complexes based on data

import numpy as np
from os import path as opath
from pathlib import Path
from sys import path as spath

# Add the parent directory to the Python path
spath.append(opath.join(Path(__file__).parent.resolve(), ".."))

from scipy.sparse import csr_matrix
from toponetx.classes import CombinatorialComplex
import torch
import trimesh

from .features import compute_feature_velocity, compute_targets

# Constant used for ablation study on object cells (do not change)
VIRTUAL_NODE_STARTING_ID: int = 9900


def compute_nodes_and_objects_positions(
    objects: list[dict], meshes: list[trimesh.Trimesh], nox4: bool
) -> tuple:
    """Compute the nodes and objets positions for all timesteps of a scene

    Parameters
    ----------
    objects : list[dict]
        Metadata of objects (ordered)
    meshes : list[trimesh.Trimesh]
        Objects' meshes (ordered)
    nox4 : bool
        True for ablation study on object cells, False for default HOPNet

    Returns
    -------
    np.ndarray
        Nodes positions (time, node_idx, 3)
    np.ndarray
        Nodes learning target acceleration (time, node_idx, 3)
    np.ndarray
        Nodes learning feature velocity (time, node_idx, 3)
    np.ndarray
        Objects positions (time, object_idx, 3)
    np.ndarray
        Objects learning target acceleration (time, object_idx, 3)
    np.ndarray
        Objects learning target velocity (time, object_idx, 3)
    CombinatorialComplex
        Complex common to all timesteps with [0, 1, 4]-rank cells defined
    list[np.ndarray]
        List containing unique node IDs of each mesh trianle (object_idx, triangle_idx, 3)
    """
    complex: CombinatorialComplex = CombinatorialComplex()

    objects_positions = np.array([obj["positions"] for obj in objects])
    objects_positions = np.moveaxis(objects_positions, 0, 1)  # (timesteps, objects, 3)
    timesteps = objects_positions.shape[0]
    objects_target_a: np.ndarray = np.zeros((timesteps, len(objects), 3))
    objects_feature_v: np.ndarray = np.zeros((timesteps, len(objects), 3))

    nodes_positions_list: list[np.ndarray] = []  # (timesteps, total_nodes, 3)
    triangles_ids: list[np.ndarray] = []  # (objects, triangles, 3)
    nodes_count: int = 0

    # For each timestep, compute the target features for each node
    for t in range(timesteps):
        scene: trimesh.Scene = trimesh.Scene()

        # For each object in the experiment
        for i, (object, og_mesh) in enumerate(zip(objects, meshes)):
            # Gather its transformation
            transformation = trimesh.transformations.quaternion_matrix(
                object["quaternions"][t]
            )
            transformation[:3, 3] = objects_positions[t, i]
            # Add the individual object to the global environment
            scene.add_geometry(og_mesh, transform=transformation)

            # Track the triangle IDs once per trajectory (at t0)
            if t == 0:
                triangles_ids.append(np.array(og_mesh.faces + nodes_count))
                nodes_count += og_mesh.vertices.shape[0]

        global_mesh = scene.to_mesh()
        nodes_positions_list.append(global_mesh.vertices)

    # Size (timesteps, nodes/objects, 3)
    nodes_positions = np.stack(nodes_positions_list, axis=0)
    nodes_target_a: np.ndarray = np.zeros((timesteps, nodes_positions.shape[1], 3))
    nodes_feature_v: np.ndarray = np.zeros((timesteps, nodes_positions.shape[1], 3))

    # Convert the scene to a single Combinatorial Complex
    # Add all nodes
    [complex.add_cell(v, rank=0) for v in range(global_mesh.vertices.shape[0])]
    # Add all edges
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

    # Add the 4-rank object cells and compute object features
    nodes_count: int = 0
    for i, (object, mesh) in enumerate(zip(objects, meshes)):
        cell = np.arange(nodes_count, nodes_count + mesh.vertices.shape[0])
        complex.add_cell(cell.tolist(), rank=4)
        nodes_count += mesh.vertices.shape[0]

        # Compute the target features for each node
        _, a = compute_targets(objects_positions[:, i, :])
        objects_target_a[:, i, :] = a

        # Compute the learning features
        feature_v = compute_feature_velocity(objects_positions[:, i, :])
        objects_feature_v[:, i, :] = feature_v

    # For each node, compute its target and learnable features
    for node_idx in range(nodes_positions.shape[1]):
        # Compute the target features for each node
        _, a = compute_targets(nodes_positions[:, node_idx, :])
        nodes_target_a[:, node_idx, :] = a

        # Compute the learning features
        feature_v = compute_feature_velocity(nodes_positions[:, node_idx, :])
        nodes_feature_v[:, node_idx, :] = feature_v

    return (
        nodes_positions,
        nodes_target_a,
        nodes_feature_v,
        objects_positions,
        objects_target_a,
        objects_feature_v,
        complex,
        triangles_ids,
    )


def compute_static_neighborhoods(
    ccc: CombinatorialComplex, device: torch.device = torch.device("cpu")
) -> tuple:
    """Compute static neighborhood matrices required by default HOPNet model.
    Static = the neighborhood matrices stay constant for all timesteps

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combination complex with all higher-order cells defined
    device : torch.device, optional
        Physical device to store the tensor, by default torch.device("cpu")

    Returns
    -------
    tuple
        Adjacency and incidence matrices
    """
    b04 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(0, 4)).to(device)
    return (b04,)


def compute_static_neighborhoods_extra(
    ccc: CombinatorialComplex, device: torch.device = torch.device("cpu")
) -> tuple:
    """Compute static neighborhood matrices required for ablation studies.
    Dynamic = the neighborhood matrices change at each timestep.

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combination complex with all higher-order cells defined
    device : torch.device, optional
        Physical device to store the tensor, by default torch.device("cpu")

    Returns
    -------
    tuple
        Adjacency and incidence matrices
    """
    a010 = _numpy_csr_to_torch_coo(ccc.adjacency_matrix(0, 1)).to(device)
    a101 = _numpy_csr_to_torch_coo(ccc.coadjacency_matrix(1, 0)).to(device)
    b01 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(0, 1)).to(device)
    b14 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(1, 4)).to(device)
    return (a010, a101, b01, b14)


def compute_dynamic_neighborhoods(
    ccc: CombinatorialComplex, device: torch.device = torch.device("cpu")
) -> tuple:
    """Compute dynamic neighborhood matrices required by default HOPNet model.
    Dynamic = the neighborhood matrices change at each timestep.

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combination complex with all higher-order cells defined
    device : torch.device, optional
        Physical device to store the tensor, by default torch.device("cpu")

    Returns
    -------
    tuple
        Adjacency and incidence matrices
    """
    collisions_count: int = len([c for c in ccc.cells if len(c) == 3])
    if collisions_count == 0:
        b02 = None
        b12 = None
        b23 = None
        b24 = None
    else:
        # Nodes-Faces incidence matrix
        b02 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(0, 2)).to(device)
        # Edges-Faces incidence matrix
        b12 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(1, 2)).to(device)
        # Faces-Collisions incidence matrix
        b23 = ccc.incidence_matrix(2, 3, sparse=False)

        # No need to convert if there are no columns
        faces = [tuple(c) for c in ccc.cells if len(c) == 3]
        collisions = [tuple(c) for c in ccc.cells if len(c) == 6]
        order = ccc.get_cell_attributes("face_0")
        b23 = _signed_incidence(b23, faces, collisions, order).to(device)
        # Faces-Objects incidence matrix
        b24 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(2, 4)).to(device)

    return (b02, b12, b23, b24)


def compute_dynamic_neighborhoods_extra(
    ccc: CombinatorialComplex, device: torch.device = torch.device("cpu")
) -> tuple:
    """Compute extra dynamic neighborhood matrices required for ablation studies.
    Dynamic = the neighborhood matrices change at each timestep.

    Parameters
    ----------
    ccc : CombinatorialComplex
        Combination complex with all higher-order cells defined
    device : torch.device, optional
        Physical device to store the tensor, by default torch.device("cpu")

    Returns
    -------
    tuple
        Extra adjacency and incidence matrices
    """
    collisions_count: int = len([c for c in ccc.cells if len(c) == 3])
    if collisions_count == 0:
        a232 = None
        b03 = None
        b13 = None
    else:
        a232 = _numpy_csr_to_torch_coo(ccc.adjacency_matrix(2, 3)).to(device)
        b03 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(0, 3)).to(device)
        b13 = _numpy_csr_to_torch_coo(ccc.incidence_matrix(1, 3)).to(device)

    return a232, b03, b13


def _numpy_csr_to_torch_coo(m) -> torch.Tensor:
    """https://discuss.pytorch.org/t/creating-a-sparse-tensor-from-csr-matrix/13658/5"""
    coo = m.tocoo()
    return torch.sparse_coo_tensor(
        torch.LongTensor(np.vstack((coo.row, coo.col))),
        torch.FloatTensor(coo.data),
        torch.Size(coo.shape),
    ).coalesce()


def _signed_incidence(
    b: csr_matrix, rows: list[tuple], cols: list[tuple], order: dict
) -> torch.Tensor:
    """Create the signed version (with values in {-1,0,+1) of an incidence matrix

    Parameters
    ----------
    b : csr_matrix
        Incidence matrix (with values in {0, +1}) as a Numpy sparse CSR matrix
    rows : list[tuple]
        Ordered IDs of cells in rows
    cols : list[tuple]
        Ordered IDs of cells in columns
    order : dict
        Dictionary of ordered tuples with 3-rank cell collisions

    Returns
    -------
    torch.Tensor
        Signed incidence matrix as a Pytorch sparse COO tensor
    """
    assert b.shape[0] == len(rows)
    assert b.shape[1] == len(cols)
    assert 2 * len(rows[0]) == len(cols[0])

    b_signed = b.copy()
    for c in range(b.shape[1]):
        col_face = cols[c]
        face_0 = order[frozenset(col_face)]
        for r in range(b.shape[0]):
            if b[r, c] != 0:
                # Check if the row is first in the cell
                if set(face_0) == set(rows[r]):
                    b_signed[r, c] = -1.0 * b[r, c]

    return torch.tensor(b_signed).to_sparse_coo()
