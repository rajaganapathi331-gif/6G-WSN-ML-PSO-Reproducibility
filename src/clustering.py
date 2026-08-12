from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _alive_indices(alive):
    return np.flatnonzero(np.asarray(alive, dtype=bool))


def leach_cluster_heads(alive, rng, fraction: float):
    idx = _alive_indices(alive)
    if len(idx) == 0:
        return np.array([], dtype=int)
    k = max(1, int(round(len(idx) * fraction)))
    k = min(k, len(idx))
    return np.sort(rng.choice(idx, size=k, replace=False))


def leach_c_cluster_heads(positions, energy, alive, fraction: float, seed: int):
    idx = _alive_indices(alive)
    if len(idx) == 0:
        return np.array([], dtype=int)
    k = max(1, int(round(len(idx) * fraction)))
    k = min(k, len(idx))

    if k == 1:
        return np.array([idx[np.argmax(energy[idx])]], dtype=int)

    km = KMeans(n_clusters=k, n_init=5, random_state=int(seed))
    labels = km.fit_predict(positions[idx])
    heads = []
    for c in range(k):
        members = idx[labels == c]
        center = km.cluster_centers_[c]
        d = np.linalg.norm(positions[members] - center, axis=1)
        en = energy[members] / max(np.max(energy[members]), 1e-12)
        score = en - 0.25 * (d / max(np.max(d), 1.0))
        heads.append(int(members[np.argmax(score)]))
    return np.array(sorted(set(heads)), dtype=int)


def proposed_cluster_heads(
    positions, energy, initial_energy, alive, predicted_drain, link_quality,
    traffic, sink, fraction: float, seed: int, weights: dict
):
    idx = _alive_indices(alive)
    if len(idx) == 0:
        return np.array([], dtype=int)
    k = max(1, int(round(len(idx) * fraction)))
    k = min(k, len(idx))

    if k == 1:
        return np.array([idx[np.argmax(energy[idx])]], dtype=int)

    km = KMeans(n_clusters=k, n_init=5, random_state=int(seed))
    labels = km.fit_predict(positions[idx])
    heads = []

    for c in range(k):
        members = idx[labels == c]
        center = km.cluster_centers_[c]
        centrality = 1.0 / (1.0 + np.linalg.norm(positions[members] - center, axis=1))
        centrality /= max(np.max(centrality), 1e-12)

        e = np.clip(energy[members] / initial_energy, 0.0, 1.0)
        lq = np.clip(link_quality[members], 0.0, 1.0)
        dr = predicted_drain[members]
        dr = dr / max(np.max(dr), 1e-12)
        tr = traffic[members]
        tr = tr / max(np.max(tr), 1e-12)

        score = (
            float(weights["residual_energy"]) * e
            + float(weights["link_quality"]) * lq
            + float(weights["centrality"]) * centrality
            - float(weights["predicted_drain"]) * dr
            - float(weights["traffic"]) * tr
        )
        heads.append(int(members[np.argmax(score)]))

    return np.array(sorted(set(heads)), dtype=int)


def assign_to_heads(positions, alive, heads):
    idx = _alive_indices(alive)
    assignment = np.full(len(alive), -1, dtype=int)
    if len(heads) == 0:
        return assignment

    head_pos = positions[heads]
    for i in idx:
        if i in heads:
            assignment[i] = int(i)
        else:
            nearest = int(np.argmin(np.linalg.norm(head_pos - positions[i], axis=1)))
            assignment[i] = int(heads[nearest])
    return assignment
