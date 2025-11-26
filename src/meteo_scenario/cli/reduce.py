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


def iter_time_batches(ds_std: xr.Dataset, vars_list, batch_t: int):
    """
    Itère sur le temps par blocs, renvoie (s, e, Xb) avec Xb shape (e-s, F).
    L'empilement se fait variable par variable pour limiter la RAM.
    """
    T = ds_std.sizes["time"]
    for s in range(0, T, batch_t):
        e = min(T, s + batch_t)
        blocks = []
        for v in vars_list:
            # (e-s, Y, X) => (e-s, Y*X)
            A = (
                ds_std[v]
                .isel(time=slice(s, e))
                .transpose("time", "latitude", "longitude")
                .values
            )
            blocks.append(A.reshape(A.shape[0], -1))
        Xb = np.concatenate(blocks, axis=1)
        yield s, e, Xb


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

    # PCA (alias --uce-pca conservé)
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--uce-pca", action="store_true", help="Alias of --use-pca")
    ap.add_argument("--components", type=int, default=15)

    # Streaming / memmap options (nouveaux, facultatifs)
    ap.add_argument(
        "--batch-time",
        type=int,
        default=512,
        help="Taille de batch en pas de temps pour le streaming.",
    )
    ap.add_argument(
        "--fit-sample-rate",
        type=float,
        default=1.0,
        help="Fraction de pas de temps utilisée pour le fit IPCA (0<r<=1).",
    )
    ap.add_argument(
        "--memmap-name",
        type=str,
        default="steps_embed.memmap",
        help="Nom du fichier memmap pour l'embedding.",
    )

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

    # --- Standardisation temporelle (anomalies) + downcast ---
    mean_t = ds.mean("time", skipna=True)
    std_t = ds.std("time", skipna=True)
    std_t = xr.where(std_t == 0, 1.0, std_t)
    ds_std = ((ds - mean_t) / std_t).astype("float32")  # réduit la RAM

    times = pd.DatetimeIndex(ds_std["time"].values)  # (T,)
    T = ds_std.sizes["time"]
    vars_list = list(ds_std.data_vars)

    print(ds_std)
    print("start streaming PCA/memmap")

    # --- Streaming + memmap ---
    # 1) Première passe: somme et compte pour la moyenne globale (centrage)
    # On évite de construire X complet.
    batch_t = max(1, int(args.batch_time))
    sum_vec = None
    n_seen = 0
    F = None
    for _, _, Xb in iter_time_batches(ds_std, vars_list, batch_t):
        if F is None:
            F = Xb.shape[1]
        if sum_vec is None:
            sum_vec = Xb.sum(axis=0, dtype=np.float64)
        else:
            sum_vec += Xb.sum(axis=0, dtype=np.float64)
        n_seen += Xb.shape[0]
    if n_seen == 0:
        raise ValueError("Dataset vide après sélection/agrégation.")
    mean = (sum_vec / float(n_seen)).astype(np.float32)  # (F,)

    # 2) Fit incrémental (optionnel) sur un échantillon temporel
    use_pca = bool(args.use_pca)
    comp = min(int(args.components), F) if use_pca else F
    ipca = None
    if use_pca:
        ipca = IncrementalPCA(n_components=comp)
        rng = np.random.RandomState(42)
        sample_rate = float(args.fit_sample_rate)
        sample_rate = 1.0 if not (0.0 < sample_rate <= 1.0) else sample_rate

        for s, e, Xb in iter_time_batches(ds_std, vars_list, batch_t):
            if sample_rate < 1.0:
                mask = rng.rand(Xb.shape[0]) < sample_rate
                if not mask.any():
                    continue
                Xc = (Xb - mean) if mean.ndim == 1 else (Xb - mean.reshape(1, -1))
                ipca.partial_fit(Xc[mask])
            else:
                Xc = (Xb - mean) if mean.ndim == 1 else (Xb - mean.reshape(1, -1))
                ipca.partial_fit(Xc)

    # 3) Transform incrémental -> memmap (et fallback sans PCA si demandé)
    steps_path = OUT / args.memmap_name
    steps_mm = np.memmap(steps_path, dtype="float32", mode="w+", shape=(T, comp))
    write_pos = 0
    for s, e, Xb in iter_time_batches(ds_std, vars_list, batch_t):
        Xc = (Xb - mean) if mean.ndim == 1 else (Xb - mean.reshape(1, -1))
        if use_pca:
            Z = ipca.transform(Xc)
        else:
            # Pas de PCA: on garde l'espace complet (potentiellement grand)
            Z = Xc
        steps_mm[s:e, :] = Z.astype("float32", copy=False)
        write_pos = e
    del steps_mm  # flush sur disque

    # Réouvre en lecture (zéro copie) pour la suite du pipeline
    steps_embed = np.memmap(steps_path, dtype="float32", mode="r", shape=(T, comp))

    print("finish pca/streaming")
    print("start windows")

    # --- Fenêtres glissantes ---
    wins = build_windows(times, args.window_hours, args.stride_hours)  # list[(s,e)]
    if not wins:
        raise ValueError("No window fits: enlarge time range or reduce --window-hours.")
    # Attention: steps_embed est un memmap 2D (T, d) -> slicing OK
    seq_list = [
        np.asarray(steps_embed[s : e + 1]) for (s, e) in wins
    ]  # (L, d) par fenêtre
    win_bounds = np.asarray(wins, dtype=int)  # (n_win, 2)

    # --- Clustering ---
    print("start clustering")
    k = args.clusters
    if args.seq_metric == "euclid":
        labels_seq, centers, window_score = cluster_sequences_euclid(seq_list, k)
        medoid_win_idx = medoid_indices_from_centroids(seq_list, labels_seq, centers)
    else:
        labels_seq, medoid_win_idx, window_score = cluster_sequences_dtw(
            seq_list,
            k,
            pam_iters=args.pam_iters,
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
    print(f" - {args.memmap_name} (embedding memmap)")
    if args.save_windows:
        print(" - windows_assignments.csv (optionnel)")


if __name__ == "__main__":
    main()
