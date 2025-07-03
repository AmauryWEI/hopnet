# File:         features.py
# Date:         2024/05/15
# Description:  Contains functions to compute learnable and target features


import numpy as np
from toponetx.classes import CombinatorialComplex
import torch

from .triangles import compute_normals, distance_triangle_triangle


def compute_feature_velocity(x: np.ndarray) -> np.ndarray:
    """Compute finite-difference velocities used as learning features

    Parameters
    ----------
    x : np.ndarray
        3D positions (timesteps, 3)

    Returns
    -------
    np.ndarray
        3D finite difference velocities (timesteps, 3) (first row is NaN)
    """
    assert x.shape[1] == 3  # (row: timestep ; columns: X, Y, Z)
    # Velocity: v_t = x_t - x_t-1
    delta_x = x[1:] - x[0:-1]
    v = np.zeros_like(x)
    v[1:] = delta_x
    v[0] = np.nan

    return v


def compute_targets(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-difference accelerations used as learning targets

    Parameters
    ----------
    x : np.ndarray
        3D positions (timesteps, 3)

    Returns
    -------
    np.ndarray
        3D finite difference accelerations (2nd order) (timesteps, 3) (first and last rows are NaN)
    """
    assert pos.shape[1] == 3  # (row: timestep ; columns: X, Y, Z)
    # Velocity: v_t = x_t+1 - x_t
    delta_x = pos[1:] - pos[0:-1]
    v = np.zeros_like(pos)
    v[:-1] = delta_x
    v[-1] = np.nan
    # Acceleration: a_t = x_t+1 - 2*x_t + x_t-1
    delta_xx = pos[2:] - 2 * pos[1:-1] + pos[0:-2]
    a = np.zeros_like(pos)
    a[1:-1] = delta_xx
    a[0] = np.nan
    a[-1] = np.nan

    return v, a


def compute_collision_features(
    t0_vert: np.ndarray, t1_vert: np.ndarray
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the collision features based on the vertices of the 2 colliding triangles

    Parameters
    ----------
    t0_vert : np.ndarray
        Vertices of triangle 0
    t1_vert : np.ndarray
        Vertices of triangle 1

    Returns
    -------
    bool
        True if the collision is sane, False otherwise
    np.ndarray
        Collision cell features (for 3-rank cell)
    np.ndarray
        Normal vector of triangle 0
    np.ndarray
        Normal vector of triangle 1
    """
    # Get the normals of both triangles (facing each other)
    t0_n, t1_n = compute_normals(t0_vert, t1_vert)

    # Compute the closest point on each of the 2 triangles
    t0_closest, t1_closest, distance = distance_triangle_triangle(t0_vert, t1_vert)
    assert t0_closest.shape == (3,)
    assert t1_closest.shape == (3,)
    assert distance >= 0.0

    # Compute the direction vector (from t0 to t1)
    if distance > 0.0:
        collision_dir = t1_closest - t0_closest
    else:
        # If objects are in direct contact
        return False, np.zeros((0,)), np.zeros((3,)), np.zeros((3,))

    # Compute the distance from each triangle node to the closest point
    t0_dist = t0_vert - t0_closest
    t1_dist = t1_vert - t1_closest
    t0_dist_norm = np.linalg.norm(t0_dist, axis=1)
    t1_dist_norm = np.linalg.norm(t1_dist, axis=1)
    t0_dist = np.hstack([t0_dist, t0_dist_norm.reshape(3, 1)])
    t1_dist = np.hstack([t1_dist, t1_dist_norm.reshape(3, 1)])

    # Sort the distances with closest first
    t0_idx = np.argsort(t0_dist[:, 3], axis=0)
    t0_dist = t0_dist[t0_idx]
    t1_idx = np.argsort(t1_dist[:, 3], axis=0)
    t1_dist = t1_dist[t1_idx]

    # Realign the normals using the collision points
    if np.dot(t1_closest - t0_closest, t0_n) < 0:
        t0_n = -1.0 * t0_n
    if np.dot(t0_closest - t1_closest, t1_n) < 0:
        t1_n = -1.0 * t1_n

    # Compile features together
    col_features = np.hstack(
        [collision_dir, distance, t0_dist.flatten(), t1_dist.flatten()]
    )

    return True, col_features, t0_n, t1_n


def get_feature_matrices(cccs: list[CombinatorialComplex], horizon: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor | None],
    list[torch.Tensor | None],
    torch.Tensor,
]:
    """Get learning features matrices for [0, 1, 2, 3, 4]-rank cells

    Parameters
    ----------
    cccs : list[CombinatorialComplex]
        Spatio-temporal combinatorial complexes
    horizon : int
        Horizon (used for safety check purposes only)

    Returns
    -------
    tuple
        Feature matrices as Pytorch float32 tensors (or list of tensors) (x0, x1, x2, x3, x4)
    """
    x0_l: list = []
    x1_l: list = []
    x2_l: list = []
    x3_l: list = []
    x4_l: list = []

    for ccc in cccs:
        # Node feature matrix x_0
        node_features_dict = ccc.get_node_attributes("features")
        features_dim_size = node_features_dict[next(iter(node_features_dict))].shape[0]
        if features_dim_size == (horizon * (3 + 1)):
            # Ablation study without center-mass distance
            pass
        elif features_dim_size == (horizon * (3 + 1) + 2 * (3 + 1)):
            # Default features
            pass
        elif features_dim_size == (horizon * (3 + 1) + 2 * (3 + 1) + (2 + 1 + 2 * 1)):
            # Ablation study without 4-rank object cells
            pass
        else:
            raise RuntimeError(f"Unepexcted node feature size: {features_dim_size}")
        x_0 = np.zeros((len(node_features_dict), features_dim_size))
        for k, node in enumerate(node_features_dict):
            x_0[k, :] = node_features_dict[node]
        assert not np.isnan(x_0).any()
        x0_l.append(torch.tensor(x_0, dtype=torch.float32))

        # Edge feature matrix x_1
        edge_features_dict = ccc.get_cell_attributes("edge_features")
        x_1 = np.zeros((len(edge_features_dict), 2 * (3 + 1)))
        for k, edge in enumerate(edge_features_dict):
            x_1[k, :] = edge_features_dict[edge]
        assert not np.isnan(x_1).any()
        x1_l.append(torch.tensor(x_1, dtype=torch.float32))

        # Faces feature matrix x_2
        faces_features_dict = ccc.get_cell_attributes("face_features")
        if len(faces_features_dict) == 0:
            x_2 = None
        else:
            columns = next(iter(faces_features_dict.values())).shape[0]
            x_2 = np.zeros((len(faces_features_dict), columns))
            for k, face in enumerate(faces_features_dict):
                x_2[k, :] = faces_features_dict[face]
            assert not np.isnan(x_2).any()
            x_2 = torch.tensor(x_2, dtype=torch.float32)
        x2_l.append(x_2)

        # Collision feature matrix x_3
        col_features_dict = ccc.get_cell_attributes("col_features")
        if len(col_features_dict) == 0:
            x_3 = None
        else:
            columns = next(iter(col_features_dict.values())).shape[0]
            x_3 = np.zeros((len(col_features_dict), columns))
            for k, cell in enumerate(col_features_dict):
                x_3[k, :] = col_features_dict[cell]
            assert not np.isnan(x_3).any()
            x_3 = torch.tensor(x_3, dtype=torch.float32)
        x3_l.append(x_3)

        # Object feature matrix x_4
        obj_features_dict = ccc.get_cell_attributes("obj_features")
        x_4 = np.zeros((len(obj_features_dict), horizon * (3 + 1) + 5))
        for k, cell in enumerate(obj_features_dict):
            x_4[k, :] = obj_features_dict[cell]
        assert not np.isnan(x_4).any()
        x4_l.append(torch.tensor(x_4, dtype=torch.float32))

    return torch.stack(x0_l), torch.stack(x1_l), x2_l, x3_l, torch.stack(x4_l)


def get_target_matrices(
    cccs: list[CombinatorialComplex],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get learning targets matrices for 0- and 4-rank cells

    Parameters
    ----------
    cccs : list[CombinatorialComplex]
        Spatio-temporal combinatorial complexes

    Returns
    -------
    tuple
        Target node and object matrices as Pytorch float32 tensors (x0, x4)
    """
    t0_l: list = []
    t4_l: list = []

    for ccc in cccs:
        # Node target matrix x_0
        node_targets_dict = ccc.get_node_attributes("target_acc")
        t_0 = np.zeros((len(node_targets_dict), 3))
        for k, node in enumerate(node_targets_dict):
            t_0[k, :] = node_targets_dict[node]
        t0_l.append(torch.tensor(t_0, dtype=torch.float32))

        # Object target matrix x_4
        obj_targets_dict = ccc.get_cell_attributes("obj_target_acc")
        t_4 = np.zeros((len(obj_targets_dict), 3))
        for k, cell in enumerate(obj_targets_dict):
            t_4[k, :] = obj_targets_dict[cell]
        t4_l.append(torch.tensor(t_4, dtype=torch.float32))

    return torch.stack(t0_l), torch.stack(t4_l)


def get_nodes_learning_masks(x4_mask: np.ndarray, b04: torch.Tensor) -> torch.Tensor:
    """Compute the learning masks for nodes & objects to avoid slow collisions"""
    x0_mask = np.matmul(x4_mask, b04.to_dense().T)

    return x0_mask.float()
