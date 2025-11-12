from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ---------- DTW multivariée (euclid per step) ----------
def dtw_distance(seqA: np.ndarray, seqB: np.ndarray) -> float:
    La, Lb = seqA.shape[0], seqB.shape[0]
    D = np.full((La + 1, Lb + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, La + 1):
        ai = seqA[i - 1]
        for j in range(1, Lb + 1):
            bj = seqB[j - 1]
            cost = float(np.linalg.norm(ai - bj))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[La, Lb])


def pairwise_dtw(seq_list: List[np.ndarray]) -> np.ndarray:
    N = len(seq_list)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = dtw_distance(seq_list[i], seq_list[j])
    return D


# ---------- Medoids: init farthest-first + PAM swaps ----------
def medoids_from_D(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    centers = [int(np.argmin(D.sum(axis=1)))]
    while len(centers) < k:
        dmin = np.min(D[:, centers], axis=1)
        nxt = int(np.argmax(dmin))
        if nxt in centers:
            break
        centers.append(nxt)
    labels = np.argmin(D[:, centers], axis=1).astype(int)
    return np.array(centers), labels


def pam_refine(
    D: np.ndarray, centers_in: np.ndarray, iters: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    centers = list(int(c) for c in centers_in)

    def assign_labels(D, centers):
        return np.argmin(D[:, centers], axis=1).astype(int)

    def total_cost(D, centers, labels):
        c = np.array(centers)
        return float(D[np.arange(D.shape[0]), c[labels]].sum())

    labels = assign_labels(D, centers)
    best_cost = total_cost(D, centers, labels)
    k = len(centers)
    N = D.shape[0]
    for _ in range(iters):
        improved = False
        for ci in range(k):
            for cand in range(N):
                if cand in centers:
                    continue
                new_centers = centers.copy()
                new_centers[ci] = cand
                new_labels = assign_labels(D, new_centers)
                new_cost = total_cost(D, new_centers, new_labels)
                if new_cost + 1e-9 < best_cost:
                    centers, labels, best_cost = new_centers, new_labels, new_cost
                    improved = True
        if not improved:
            break
    return np.array(centers), labels


# ---------- Euclidean (MiniBatchKMeans on flattened sequences) ----------
def cluster_sequences_euclid(
    seq_list: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns labels, centers (centroids in flatten space), window_score (0..1 per window).
    """
    seq_flat = np.array([seq.ravel() for seq in seq_list])  # (N, L*d)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=512)
    labels = km.fit_predict(seq_flat)
    centers = km.cluster_centers_
    d_to_center = np.linalg.norm(seq_flat - centers[labels], axis=1)

    window_score = np.zeros(len(seq_flat), dtype=float)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d = d_to_center[idx]
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = np.ones_like(d)
        window_score[idx] = sc
    return labels, centers, window_score


def medoid_indices_from_centroids(
    seq_list: List[np.ndarray], labels: np.ndarray, centers: np.ndarray
) -> Dict[int, int]:
    """Return medoid index per cluster as closest window to centroid in flatten space."""
    seq_flat = np.array([s.ravel() for s in seq_list])
    med = {}
    k = centers.shape[0]
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        d = np.linalg.norm(seq_flat[idx] - centers[c][None, :], axis=1)
        med[c] = int(idx[np.argmin(d)])
    return med


# ---------- DTW + PAM ----------
def cluster_sequences_dtw(
    seq_list: List[np.ndarray], k: int, pam_iters: int = 8
) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    """
    Returns labels, medoid_win_idx (dict cluster->index), window_score (0..1).
    """
    D = pairwise_dtw(seq_list)
    centers0, labels = medoids_from_D(D, k)
    centersA, labels = pam_refine(D, centers0, iters=pam_iters)
    medoid_win_idx = {c: int(centersA[c]) for c in range(k)}

    # score = 1 - normalized distance to medoid (per cluster)
    N = len(seq_list)
    window_score = np.zeros(N, dtype=float)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        ref = medoid_win_idx[c]
        d = D[idx, ref]
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = np.ones_like(d)
        window_score[idx] = sc
    return labels, medoid_win_idx, window_score
