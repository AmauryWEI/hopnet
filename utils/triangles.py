# File:         triangles.py
# Date:         2024/05/15
# Description:  Contains functions to compute properties on mesh triangle faces

import numpy as np


def compute_normals(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute facing normals of two triangles 3D P and Q

    Parameters
    ----------
    p : np.ndarray
        3D coordinates of triangle P defined by three points (dim 0: point, dim 1: axis)
    q : np.ndarray
        3D coordinates of triangle Q defined by three points (dim 0: point, dim 1: axis)

    Returns
    -------
    np.ndarray
        Normal vector of triangle P facing triangle Q
    np.ndarray
        Normal vector of triangle Q facing triangle P
    """
    # Compute triangle 0 unit normal vector
    t0_ab = p[1, :] - p[0, :]
    t0_ac = p[2, :] - p[0, :]
    t0_n = np.cross(t0_ab, t0_ac)
    t0_n = t0_n / np.linalg.norm(t0_n)

    # Compute triangle 1 unit normal vector
    t1_ab = q[1, :] - q[0, :]
    t1_ac = q[2, :] - q[0, :]
    t1_n = np.cross(t1_ab, t1_ac)
    t1_n = t1_n / np.linalg.norm(t1_n)

    # Compute barycenters of faces
    t0_bary = np.mean(p, axis=0)
    t1_bary = np.mean(q, axis=0)

    # Re-adjust the normals (if needed) to make sure they face each other
    # Source: https://gamedev.stackexchange.com/a/185581
    if np.dot(t1_bary - t0_bary, t0_n) < 0:
        t0_n = -1.0 * t0_n
    if np.dot(t0_bary - t1_bary, t1_n) < 0:
        t1_n = -1.0 * t1_n

    return t0_n, t1_n


def distance_triangle_triangle(
    p: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the closest distance between two triangles  and q

    Source: https://github.com/NVIDIA-Omniverse/PhysX/blob/a2af52eb6a2532bd2bc583ef8ead9c81c9222af1/physx/source/geomutils/src/distance/GuDistanceTriangleTriangle.cpp

    Parameters
    ----------
    p : np.ndarray
        3D coordinates of triangle P defined by three points (dim 0: point, dim 1: axis)
    q : np.ndarray
        3D coordinates of triangle Q defined by three points (dim 0: point, dim 1: axis)

    Returns
    -------
    np.ndarray
        Coordinates of closest point on triangle P
    np.ndarray
        Coordinates of closest point on triangle Q
    float
        Shorted distance between triangles P and Q
    """

    sv: np.ndarray = np.zeros((3, 3))
    sv[0, :] = p[1, :] - p[0, :]
    sv[1, :] = p[2, :] - p[1, :]
    sv[2, :] = p[0, :] - p[2, :]

    tv: np.ndarray = np.zeros((3, 3))
    tv[0, :] = q[1, :] - q[0, :]
    tv[1, :] = q[2, :] - q[1, :]
    tv[2, :] = q[0, :] - q[2, :]

    mindd: float = np.inf
    shown_disjoint = False

    for i in range(0, 3):
        for j in range(0, 3):
            cp, cq = _edge_edge_dist(p[i, :], sv[i, :], q[j, :], tv[j, :])
            v = cq - cp
            dd = v.dot(v)

            if dd <= mindd:
                minP = cp
                minQ = cq
                mindd = dd

                id = i + 2
                if id >= 3:
                    id -= 3
                z0 = p[id] - cp
                a = z0.dot(v)

                id = j + 2
                if id >= 3:
                    id -= 3
                z1 = q[id] - cq
                b = z1.dot(v)

                if (a <= 0.0) and (b >= 0.0):
                    return cp, cq, v.dot(v)
                if a <= 0.0:
                    a = 0.0
                elif b > 0.0:
                    b = 0.0
                if (mindd - a + b) > 0.0:
                    shown_disjoint = True

    sn = np.cross(sv[0], sv[1])
    snl = sn.dot(sn)

    if snl > 1e-15:
        tp = np.array(
            [
                (p[0] - q[0]).dot(sn),
                (p[0] - q[1]).dot(sn),
                (p[0] - q[2]).dot(sn),
            ]
        )
        index = -1
        if (tp[0] > 0.0) and tp[1] > 0.0 and tp[2] > 0.0:
            index = 0 if tp[0] < tp[1] else 1
            if tp[2] < tp[index]:
                index = 2
        elif tp[0] < 0.0 and tp[1] < 0.0 and tp[2] < 0.0:
            index = 0 if tp[0] > tp[1] else 1
            if tp[2] > tp[index]:
                index = 2

        if index >= 0:
            shown_disjoint = True
            q_index = q[index]
            V = q_index - p[0]
            Z = np.cross(sn, sv[0])
            if V.dot(Z) > 0.0:
                V = q_index - p[1]
                Z = np.cross(sn, sv[1])
                if V.dot(Z) > 0.0:
                    V = q_index - p[2]
                    Z = np.cross(sn, sv[2])
                    if V.dot(Z) > 0.0:
                        cp = q_index + sn * tp[index] / snl
                        cq = q_index
                        return cp, cq, np.linalg.norm(cp - cq)

    tn = np.cross(tv[0], tv[1])
    tnl = tn.dot(tn)

    if tnl > 1e-15:
        sp = np.array(
            [
                (q[0] - p[0]).dot(tn),
                (q[0] - p[1]).dot(tn),
                (q[0] - p[2]).dot(tn),
            ]
        )
        index = -1
        if sp[0] > 0.0 and sp[1] > 0.0 and sp[2] > 0.0:
            index = 0 if sp[0] < sp[1] else 1
            if sp[2] < sp[index]:
                index = 2
        elif sp[0] < 0.0 and sp[1] < 0.0 and sp[2] < 0.0:
            index = 0 if sp[0] > sp[1] else 1
            if sp[2] > sp[index]:
                index = 2

        if index >= 0:
            shown_disjoint = True
            p_index = p[index]
            V = p_index - q[0]
            Z = np.cross(tn, tv[0])
            if V.dot(Z) > 0.0:
                V = p_index - q[1]
                Z = np.cross(tn, tv[1])
                if V.dot(Z) > 0.0:
                    V = p_index - q[2]
                    Z = np.cross(tn, tv[2])
                    if V.dot(Z) > 0.0:
                        cp = p_index
                        cq = p_index + tn * sp[index] / tnl
                        return cp, cq, np.linalg.norm(cp - cq)

    if shown_disjoint:
        cp = minP
        cq = minQ
        return cp, cq, mindd
    else:
        return cp, cq, 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return np.min([high, np.max([low, value])])


def _edge_edge_dist(
    p: np.ndarray, a: np.ndarray, q: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the closest points between two 3D segments, defined by their origin
    (p or q) and their vector (a or b)

    Parameters
    ----------
    p : np.ndarray
        Point P of segment (P, A)
    a : np.ndarray
        Vector A of segment (P, A)
    q : np.ndarray
        Point Q of segment (Q, B)
    b : np.ndarray
        Vector Q of segment (Q, B)

    Returns
    -------
    np.ndarray
        Coordinates of closest point on (P, A)
    np.ndarray
        Coordinates of closest point on (Q, B)
    """
    T = q - p
    a_dot_a = a.dot(a)
    b_dot_b = b.dot(b)
    a_dot_b = a.dot(b)
    a_dot_t = a.dot(T)
    b_dot_t = b.dot(T)

    denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b
    if denom != 0.0:
        # Clamp the result so t in on the segment (P, A)
        t = _clamp((a_dot_t * b_dot_b - b_dot_t * a_dot_b) / denom, 0.0, 1.0)
    else:
        t = 0.0

    # Find u for point on (Q, B) closest to point at t
    if b_dot_b != 0.0:
        u = (t * a_dot_b - b_dot_t) / b_dot_b
        # If u is on segment (Q, B), t and u are the closest points, otherwise clamp u and recompute t
        if u < 0.0:
            u = 0.0
            if a_dot_a != 0.0:
                t = _clamp(a_dot_t / a_dot_a, 0.0, 1.0)
            else:
                t = 0.0
        elif u > 1.0:
            u = 1.0
            if a_dot_a != 0.0:
                t = _clamp((a_dot_b + a_dot_t) / a_dot_a, 0.0, 1.0)
            else:
                t = 0.0
    else:
        u = 0.0
        if a_dot_a != 0.0:
            t = _clamp(a_dot_t / a_dot_a, 0.0, 1.0)
        else:
            t = 0.0

    return p + t * a, q + u * b
