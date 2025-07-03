# File:         collisions.py
# Date:         2024/05/15
# Description:  Contains functions to compute collisions between objects

from copy import deepcopy

import numpy as np
import trimesh


def compute_collisions(
    obj_pos: list[np.ndarray] | np.ndarray,
    obj_quat: list[np.ndarray] | np.ndarray,
    meshes: list[trimesh.Trimesh],
    collision_radius: float,
) -> list[dict]:
    """Compute colliding faces between objects for a single timestep

    Parameters
    ----------
    obj_pos : np.ndarray
        Objects' positions ; shape [objs_count, 3]
    obj_quat : np.ndarray
        Objects' quaternions ; shape [objs_count, 4]
    meshes : list[trimesh.Trimesh]
        Original objects' meshes
    collision_radius : float
        Threshold distance between faces to assume a collision

    Returns
    -------
    list[dict]
        Detected collisions
    """
    tmp_meshes: list[trimesh.Trimesh] = []

    # Apply the rotation and translation to each mesh
    for idx in range(len(meshes)):
        tmp_mesh = meshes[idx].copy()

        # Apply the initial rotation, then translation to the object
        pos: np.ndarray = obj_pos[idx]
        quat: np.ndarray = obj_quat[idx]
        tmp_mesh.apply_transform(trimesh.transformations.quaternion_matrix(quat))
        tmp_mesh.apply_translation(pos)

        tmp_meshes.append(tmp_mesh)

    # For each object, find the closest point in other meshes to the current mesh
    tmp_collisions: list = []
    for i, source in enumerate(tmp_meshes):
        for k, target in enumerate(tmp_meshes):
            # Do not consider when objects are the same
            if i == k:
                continue

            # Find the closest points on mesh i to the vertices of mesh k
            _, distances, triangle_ids = trimesh.proximity.closest_point(
                source, target.vertices
            )

            # Check if there are any points of mesh k close to mesh i
            indices = np.atleast_1d(
                np.argwhere(distances <= collision_radius).squeeze()
            )

            # If the 2 meshes are not close enough, continue to the next object
            if not indices.any():
                continue

            # Connect each triangle/face of mesh i with the faces of mesh k the closest point
            pairs: list = []
            for p_idx in indices:
                # Get the triangle of mesh i
                triangle_id_of_i = triangle_ids[p_idx]
                # Find which faces of mesh k contain the close node
                face_ids_of_k = np.argwhere(target.faces == p_idx)[:, 0]

                # Create a pair between the face of i and each close face of k
                for face_id_of_k in face_ids_of_k:
                    pairs.append((int(triangle_id_of_i), int(face_id_of_k)))

            # Remove duplicate pairs (does not fully remove inverted duplicates)
            tmp_collisions.append(
                {
                    "obj0": i,
                    "obj1": k,
                    "pairs": list(dict.fromkeys(pairs)),
                }
            )

    return _remove_duplicates(tmp_collisions)


def compute_collisions_timeseries(
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    collision_radius: float,
) -> dict:
    """Compute the collisions between mesh triangles for alll timesteps of a sample

    Parameters
    ----------
    objects : list[dict]
        Objects metdata
    meshes : list[trimesh.Trimesh]
        Original meshes corresponding to objects
    collision_radius : float
        Threshold distance between faces to assume a collision

    Returns
    -------
    dict
        Dictionary with timesteps as keys and collisions as values
    """
    assert collision_radius > 0
    assert len(objects) == len(meshes)

    timesteps: int = len(objects[0]["positions"])
    collisions: dict = {}

    for t in range(timesteps):
        # Build the objects' positions and quaternions for this timestep
        obj_pos = [np.array(obj["positions"][t]) for obj in objects]
        obj_quat = [np.array(obj["quaternions"][t]) for obj in objects]
        collisions.update(
            {str(t): compute_collisions(obj_pos, obj_quat, meshes, collision_radius)}
        )

    return collisions


def _remove_duplicates(collisions: list[dict]) -> list[dict]:
    """Prune duplicates inside a list of collisions

    Parameters
    ----------
    collisions : list[dict]
        Detected collisions (containing duplicates or not)

    Returns
    -------
    list[dict]
        Pruned collisions (without any duplicates)
    """
    # If no collisions detected
    if len(collisions) == 0:
        return []

    clean_collisions: list[dict] = deepcopy(collisions)

    # For each collision, find if its opposite exist
    for c in collisions:
        obj0 = c["obj0"]
        obj1 = c["obj1"]

        # Check if this exact collision has already been removed by the algorithm
        idx = next(
            (
                i
                for (i, d) in enumerate(clean_collisions)
                if d["obj0"] == obj0 and d["obj1"] == obj1
            ),
            None,
        )
        if idx is None:
            continue

        # Find the opposite collision set
        counter_c = next(
            (d for d in clean_collisions if d["obj0"] == obj1 and d["obj1"] == obj0),
            None,
        )
        # If no opposite collisions were detected, try for the next one
        if counter_c is None:
            continue

        # If a counter collision was detected, combine the two into one
        counter_pairs = counter_c["pairs"]
        inverted_pairs = [p[::-1] for p in counter_pairs]

        # Combine the pairs into this collision
        clean_pairs = list(dict.fromkeys(c["pairs"] + inverted_pairs))

        # Remove the opposite collisions
        clean_collisions.remove(counter_c)

        # Update the clean collisions
        clean_collisions[idx]["pairs"] = clean_pairs

    return clean_collisions
