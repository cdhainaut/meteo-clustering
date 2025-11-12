from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import IncrementalPCA

from ..io import open_normalize, export_grib_from_ds
from ..windows import build_windows
from ..clustering import (
    cluster_sequences_euclid,
    medoid_indices_from_centroids,
    cluster_sequences_dtw,
)
from ..export import majority_vote_labels, attach_time_labels_scores


def main():
    ap = argparse.ArgumentParser(
        description="Sequence clustering -> cluster weights + medoids + time votes (+ optional windows dump)."
    )
    ap.add_argument("grib", type=str, help="Path to merged GRIB/NetCDF")
    ap.add_argument(
        "--vars", type=str, default="u10,v10", help="Comma-separated variables to use."
    )
    ap.add_argument(
        "--coarsen", type=int, default=0, help="Spatial coarsen factor (0=off)."
    )
    ap.add_argument(
        "--time-agg",
        type=str,
        default="",
        help="Resample frequency (e.g., '6H'); empty=off.",
    )
    ap.add_argument("--window-hours", type=int, default=72)
    ap.add_argument("--stride-hours", type=int, default=24)
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--seq-metric", choices=["euclid", "dtw"], default="euclid")
    ap.add_argument("--pam-iters", type=int, default=8)

    # PCA (avec alias --uce-pca pour compat avec ta commande existante)
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--uce-pca", action="store_true", help="Alias of --use-pca")
    ap.add_argument("--components", type=int, default=15)

    # Sorties
    ap.add_argument("--out", type=str, default="reduced_out")
    ap.add_argument(
        "--save-windows",
        action="store_true",
        help="Optionnel: dump per-window (windows_assignments.csv).",
    )

    args = ap.parse_args()
    if args.uce_pca:
        args.use_pca = True

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Chargement + sélection des variables ---
    ds = open_normalize(args.grib)
    keep = [v.strip() for v in args.vars.split(",") if v.strip()]
    for v in keep:
        if v not in ds:
            raise KeyError(f"Var '{v}' not in dataset. Available: {list(ds.data_vars)}")
    ds = ds[keep].sortby(["latitude", "longitude", "time"])

    # Optionnel: coarsen spatial / agrég. temporelle
    if args.coarsen and args.coarsen > 1:
        ds = ds.coarsen(
            latitude=args.coarsen, longitude=args.coarsen, boundary="trim"
        ).mean()
    if args.time_agg:
        ds = ds.resample(time=args.time_agg).mean()

    # --- Standardisation temporelle (anomalies) ---
    mean_t = ds.mean("time", skipna=True)
    std_t = ds.std("time", skipna=True)
    std_t = xr.where(std_t == 0, 1.0, std_t)
    ds_std = (ds - mean_t) / std_t

    print(ds_std)
    print("start aplatissement")

    times = pd.DatetimeIndex(ds_std["time"].values)  # (T,)
    # Aplatissement par pas de temps -> 2D (T, F)
    blocks = []
    for v in ds_std.data_vars:
        A = ds_std[v].transpose("time", "latitude", "longitude").values  # (T, Y, X)
        blocks.append(A.reshape(A.shape[0], -1))
    X = np.concatenate(blocks, axis=1)  # (T, F)
    print("finish aplatissement")
    print("start pca")

    # --- PCA optionnelle ---
    if args.use_pca:
        Xc = X - X.mean(axis=0, keepdims=True)
        ipca = IncrementalPCA(n_components=min(args.components, X.shape[1]))
        batch = max(1, min(1024, Xc.shape[0]))
        for i in range(0, Xc.shape[0], batch):
            ipca.partial_fit(Xc[i : i + batch])
        steps_embed = np.vstack(
            [ipca.transform(Xc[i : i + batch]) for i in range(0, Xc.shape[0], batch)]
        )
    else:
        steps_embed = X

    print("finish pca")
    # --- Fenêtres glissantes ---
    wins = build_windows(times, args.window_hours, args.stride_hours)  # list[(s,e)]
    if not wins:
        raise ValueError("No window fits: enlarge time range or reduce --window-hours.")
    seq_list = [steps_embed[s : e + 1] for (s, e) in wins]  # (L, d) par fenêtre
    win_bounds = np.asarray(wins, dtype=int)  # (n_win, 2)

    # --- Clustering ---
    print("start clustering")
    k = args.clusters
    if args.seq_metric == "euclid":
        labels_seq, centers, window_score = cluster_sequences_euclid(seq_list, k)
        medoid_win_idx = medoid_indices_from_centroids(seq_list, labels_seq, centers)
    else:
        labels_seq, medoid_win_idx, window_score = cluster_sequences_dtw(
            seq_list, k, pam_iters=args.pam_iters
        )
    print("finish clustering")
    labels_seq = np.asarray(labels_seq, dtype=int)
    window_score = np.asarray(window_score, dtype=float)

    # --- Poids par cluster (+ version pondérée par la qualité de fenêtre) ---
    counts = np.bincount(labels_seq, minlength=k).astype(int)
    weight_frac = (
        counts / counts.sum() if counts.sum() > 0 else np.zeros(k, dtype=float)
    )
    wcounts = np.bincount(labels_seq, weights=window_score, minlength=k)
    wfrac = wcounts / wcounts.sum() if wcounts.sum() > 0 else np.zeros(k, dtype=float)

    # --- Résumé clusters: poids + médoïdes (une ligne par cluster) ---
    summary_rows = []
    for c in range(k):
        mw = int(medoid_win_idx[c])
        si, ei = win_bounds[mw]
        summary_rows.append(
            {
                "cluster": c,
                "weight_count": int(counts[c]),
                "weight_frac": float(weight_frac[c]),
                "weight_count_weighted": float(np.round(wcounts[c], 3)),
                "weight_frac_weighted": float(np.round(wfrac[c], 6)),
                "medoid_window_index": mw,
                "medoid_score": float(window_score[mw]),
                "medoid_start_time": pd.Timestamp(times[si]),
                "medoid_end_time": pd.Timestamp(times[ei]),
            }
        )
    pd.DataFrame(summary_rows).sort_values("cluster").to_csv(
        OUT / "cluster_summary.csv", index=False
    )

    # --- Vote au pas de temps: label + fraction de vote (confiance) ---
    labels_time, time_vote_frac = majority_vote_labels(wins, labels_seq, T=len(times))
    pd.DataFrame(
        {"time": times, "cluster_id": labels_time, "vote_frac": time_vote_frac}
    ).to_csv(OUT / "time_cluster_map.csv", index=False)

    # --- Enrichir le Dataset et exporter le GRIB (pour recoller au temps ensuite) ---
    ds_with = attach_time_labels_scores(
        ds, labels_time, time_vote_frac
    )  # garde 'cluster_id' + 'cluster_score' dans DS
    export_grib_from_ds(ds_with, OUT / "original_with_cluster_id.grib2")

    # --- Optionnel: dump détaillé par fenêtre (désactivé par défaut) ---
    if args.save_windows:
        rows = []
        for w, (si, ei) in enumerate(win_bounds):
            rows.append(
                {
                    "window_index": int(w),
                    "start_index": int(si),
                    "end_index": int(ei),
                    "start_time": pd.Timestamp(times[si]),
                    "end_time": pd.Timestamp(times[ei]),
                    "cluster": int(labels_seq[w]),
                    "window_score": float(window_score[w]),
                }
            )
        pd.DataFrame(rows).sort_values(
            ["cluster", "window_score"], ascending=[True, False]
        ).to_csv(OUT / "windows_assignments.csv", index=False)

    print(f"[DONE] Outputs in: {OUT}")
    print(" - cluster_summary.csv  (poids + médoïdes)")
    print(" - time_cluster_map.csv (vote_frac)")
    print(" - original_with_cluster_id.grib2")
    if args.save_windows:
        print(" - windows_assignments.csv (optionnel)")


if __name__ == "__main__":
    main()
