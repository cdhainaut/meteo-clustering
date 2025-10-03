#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Séquences synoptiques + étiquette de cluster par pas de temps + SCORES
# + Post-processing: plots au point (lat,lon) pour valider la qualité (u10 & v10)
#
# - cluster_score (windows): représentativité de la fenêtre dans son cluster (0..1)
# - cluster_score (time): force du vote majoritaire (0..1)
# - Exporte les scores dans les GRIB2 (via CDO) + CSV
# - Optionnel: diag au point (lat,lon): un panel par cluster (u10 & v10), toutes séquences, médonoïde, mean±std

import argparse
from pathlib import Path
import shutil, tempfile, os

import nomad.numpy as np
import numpy as npx  # NumPy classique (where/argmin/percentiles)
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans


# ---------- utils ----------
def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def export_grib_from_ds(ds: xr.Dataset, out_grib: Path) -> bool:
    tmpdir = Path(tempfile.mkdtemp(prefix="wasp_export_"))
    tmp_nc = tmpdir / "tmp.nc"
    ds.to_netcdf(tmp_nc)
    if have("cdo"):
        cmd = f'cdo -f grb2 copy "{tmp_nc}" "{out_grib}"'
        ret = os.system(cmd)
        ok = (ret == 0) and out_grib.exists()
        try:
            tmp_nc.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass
        if ok:
            return True
        print(f"[WARN] CDO failed (code {ret}). Kept NetCDF fallback.")
    else:
        print("[WARN] 'cdo' not found. Keeping NetCDF and printing conversion hint.")
    fallback_nc = out_grib.with_suffix(".nc")
    ds.to_netcdf(fallback_nc)
    print(f"→ Convert later: cdo -f grb2 copy '{fallback_nc}' '{out_grib}'")
    return False


def dtw_distance(seqA: npx.ndarray, seqB: npx.ndarray) -> float:
    # DTW multivariée (euclid par pas). seq: (L, D)
    La, Lb = seqA.shape[0], seqB.shape[0]
    D = npx.full((La + 1, Lb + 1), npx.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, La + 1):
        ai = seqA[i - 1]
        for j in range(1, Lb + 1):
            bj = seqB[j - 1]
            cost = float(npx.linalg.norm(ai - bj))
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[La, Lb])


def medoids_from_distance_matrix(D: npx.ndarray, k: int):
    # init farthest-first + simple assign (pas de swaps)
    n = D.shape[0]
    centers = [int(npx.argmin(D.sum(axis=1)))]
    while len(centers) < k:
        dmin = npx.min(D[:, centers], axis=1)
        nxt = int(npx.argmax(dmin))
        if nxt in centers:
            break
        centers.append(nxt)
    dmin = npx.min(D[:, centers], axis=1)
    labels = npx.argmin(D[:, centers], axis=1)
    return npx.array(centers), labels.astype(int)


def pam_refine(D: npx.ndarray, centers_in, iters: int = 10):
    # PAM swaps (améliore les centres)
    centers = list(int(c) for c in centers_in)

    def assign_labels(D, centers):
        return npx.argmin(D[:, centers], axis=1).astype(int)

    def total_cost(D, centers, labels):
        c = npx.array(centers)
        return float(D[npx.arange(D.shape[0]), c[labels]].sum())

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
    return npx.array(centers), labels


# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Séquences synoptiques + export GRIB avec labels & scores par time-step (+ plots au point lat/lon)"
)
parser.add_argument("grib", type=str, help="Chemin GRIB/NetCDF (cfgrib si grib)")
parser.add_argument("--vars", type=str, default="u10,v10")
parser.add_argument("--lon", type=float, nargs=2, default=(-80, 0))
parser.add_argument("--lat", type=float, nargs=2, default=(20, 60))
parser.add_argument("--time", type=str, nargs=2, default=("2020-01-01", "2020-12-30"))
parser.add_argument("--coarsen", type=int, default=6)
parser.add_argument("--time-agg", type=str, default="12H")
parser.add_argument("--use-pca", action="store_true")
parser.add_argument("--components", type=int, default=15)
parser.add_argument("--window-hours", type=int, default=24)
parser.add_argument("--stride-hours", type=int, default=24)
parser.add_argument("--clusters", type=int, default=6)
parser.add_argument(
    "--seq-metric", type=str, choices=["euclid", "dtw"], default="euclid"
)
parser.add_argument(
    "--pam-iters", type=int, default=8, help="Itérations PAM swaps (DTW)"
)

# --- post-processing au point ---
parser.add_argument(
    "--probe-lat", type=float, default=None, help="Latitude du point diag (optional)"
)
parser.add_argument(
    "--probe-lon", type=float, default=None, help="Longitude du point diag (optional)"
)
parser.add_argument(
    "--plot-out",
    type=str,
    default="probe_clusters.png",
    help="Nom du PNG de sortie pour le diag",
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Désactiver la figure même si probe-lat/lon fournis",
)

parser.add_argument("--out", type=str, default="seq_grib")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Load & subset ----------
ds = xr.open_dataset(args.grib)
if "time" not in ds.dims and "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})
if "time" not in ds.dims and "forecast_time" in ds.coords:
    ds = ds.rename({"forecast_time": "time"})
if "time" not in ds.dims:
    raise ValueError("No time dimension found.")

keep = [v.strip() for v in args.vars.split(",") if v.strip()]
for v in keep:
    if v not in ds:
        raise KeyError(f"Var '{v}' not found. Available: {list(ds.data_vars)}")
ds = ds[keep]

lat_name = (
    "latitude" if "latitude" in ds.dims else ("lat" if "lat" in ds.dims else None)
)
lon_name = (
    "longitude" if "longitude" in ds.dims else ("lon" if "lon" in ds.dims else None)
)
if lat_name is None or lon_name is None:
    raise ValueError("Missing lat/lon dims.")

ds = ds.sortby([lat_name, lon_name, "time"])
ds = ds.sel(
    **{
        lon_name: slice(args.lon[0], args.lon[1]),
        lat_name: slice(args.lat[0], args.lat[1]),
    },
    time=slice(args.time[0], args.time[1]),
)

if args.coarsen and args.coarsen > 1:
    ds = ds.coarsen(
        {lat_name: args.coarsen, lon_name: args.coarsen}, boundary="trim"
    ).mean()
if args.time_agg != "":
    ds = ds.resample(time=args.time_agg).mean()

# ---------- anomalies + flatten per timestep ----------
mean_t = ds.mean("time", skipna=True)
std_t = ds.std("time", skipna=True)
std_t = xr.where(std_t == 0, 1.0, std_t)
ds_std = (ds - mean_t) / std_t

times = pd.DatetimeIndex(ds_std["time"].values)

blocks = []
for v in ds_std.data_vars:
    A = ds_std[v].transpose("time", lat_name, lon_name).values
    blocks.append(A.reshape(A.shape[0], -1))
X = npx.concatenate(blocks, axis=1)  # (T, F)

# ---------- optional PCA (per-step embedding) ----------
if args.use_pca:
    Xc = X - X.mean(axis=0, keepdims=True)  # center only
    ipca = IncrementalPCA(n_components=min(args.components, X.shape[1]))
    batch = max(1, min(1024, Xc.shape[0]))
    for i in range(0, Xc.shape[0], batch):
        ipca.partial_fit(Xc[i : i + batch])
    steps_embed = npx.vstack(
        [ipca.transform(Xc[i : i + batch]) for i in range(0, Xc.shape[0], batch)]
    )  # (T, d)
else:
    steps_embed = X

# ---------- build sliding windows (indices) ----------
if len(times) >= 2:
    dt_hours = float((times[1] - times[0]).total_seconds()) / 3600.0
else:
    dt_hours = args.stride_hours
win_len = max(1, int(round(args.window_hours / dt_hours)))
stride = max(1, int(round(args.stride_hours / dt_hours)))

wins = []
for s in range(0, len(times) - win_len + 1, stride):
    e = s + win_len - 1
    wins.append((s, e))
if not wins:
    raise ValueError("No window fits: enlarge time range or reduce window-hours.")

# séquences = empilement des embeddings (L,d) pour la distance
seq_list = [steps_embed[s : e + 1] for (s, e) in wins]

# ---------- clustering de séquences + window scores ----------
k = args.clusters
if args.seq_metric == "euclid":
    seq_flat = npx.array([seq.ravel() for seq in seq_list])  # (N, L*d)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=512)
    labels_seq = km.fit_predict(seq_flat)

    centers = km.cluster_centers_
    d_to_center = npx.linalg.norm(seq_flat - centers[labels_seq], axis=1)

    window_score = npx.zeros(len(seq_flat), dtype=float)
    for c in range(k):
        idx = npx.where(labels_seq == c)[0]
        if idx.size == 0:
            continue
        d = d_to_center[idx]
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = npx.ones_like(d)
        window_score[idx] = sc

    medoid_win_idx = {}
    for c in range(k):
        idx = npx.where(labels_seq == c)[0]
        if idx.size == 0:
            continue
        centroid = centers[c][None, :]
        d = npx.linalg.norm(seq_flat[idx] - centroid, axis=1)
        medoid_win_idx[c] = idx[npx.argmin(d)]

else:
    # DTW: matrice de distances O(N^2)
    N = len(seq_list)
    D = npx.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = dtw_distance(seq_list[i], seq_list[j])

    centers0, labels_seq = medoids_from_distance_matrix(D, k)
    centersA, labels_seq = pam_refine(D, centers0, iters=args.pam_iters)
    medoid_win_idx = {c: int(centersA[c]) for c in range(k)}

    # scores par fenêtre = 1 - (d / max_d_cluster) à leur médonoïde
    window_score = npx.zeros(N, dtype=float)
    for c in range(k):
        idx = npx.where(labels_seq == c)[0]
        if idx.size == 0:
            continue
        ref = medoid_win_idx[c]
        ref_vec = seq_list[int(ref)]
        d = npx.array([dtw_distance(seq_list[i], ref_vec) for i in idx], dtype=float)
        d_min, d_max = float(d.min()), float(d.max())
        if d_max > d_min:
            sc = 1.0 - (d - d_min) / (d_max - d_min)
        else:
            sc = npx.ones_like(d)
        window_score[idx] = sc

# ---------- vote majoritaire + time-score ----------
cover_lists = [[] for _ in range(len(times))]
for w_i, (s, e) in enumerate(wins):
    c = int(labels_seq[w_i])
    for t in range(s, e + 1):
        cover_lists[t].append(c)

labels_time = npx.zeros(len(times), dtype=int)
time_score = npx.zeros(len(times), dtype=float)  # proportion du gagnant
for t, lst in enumerate(cover_lists):
    if len(lst) == 0:
        labels_time[t] = -1
        time_score[t] = 0.0
    else:
        vals, counts = npx.unique(npx.array(lst), return_counts=True)
        imax = int(npx.argmax(counts))
        labels_time[t] = int(vals[imax])
        time_score[t] = float(counts[imax]) / float(len(lst))

# ---------- export GRIB: (1) fenêtres médonoïdes + score ----------
OUT.mkdir(parents=True, exist_ok=True)
rows = []
n_grib = 0
for c, w_i in sorted(medoid_win_idx.items()):
    s, e = wins[w_i]
    rows.append(
        {
            "cluster": int(c),
            "score": float(window_score[w_i]),
            "start_index": int(s),
            "end_index": int(e),
            "start_time": times[s].isoformat(),
            "end_time": times[e].isoformat(),
        }
    )
    sub = ds.isel(time=slice(s, e + 1))
    T = sub.sizes["time"]
    cluster_vec = npx.full(T, int(c), dtype=int)
    score_vec = npx.full(T, float(window_score[w_i]), dtype=float)
    cluster_da = xr.DataArray(
        npx.broadcast_to(
            cluster_vec[:, None, None], (T, sub.sizes[lat_name], sub.sizes[lon_name])
        ),
        coords={"time": sub["time"], lat_name: sub[lat_name], lon_name: sub[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_id",
        attrs={"long_name": "sequence cluster label (window)", "units": "1"},
    )
    score_da = xr.DataArray(
        npx.broadcast_to(
            score_vec[:, None, None], (T, sub.sizes[lat_name], sub.sizes[lon_name])
        ),
        coords={"time": sub["time"], lat_name: sub[lat_name], lon_name: sub[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_score",
        attrs={"long_name": "representativeness score (0..1)", "units": "1"},
    )
    sub_with = sub.copy()
    sub_with["cluster_id"] = cluster_da
    sub_with["cluster_score"] = score_da
    out_grib = OUT / f"scenario_seq_cluster{c}.grib2"
    if export_grib_from_ds(sub_with, out_grib):
        n_grib += 1

pd.DataFrame(rows).to_csv(OUT / "scenarios_sequences_summary.csv", index=False)
print(f"[OK] Windows GRIB2 (with score): {n_grib}/{len(medoid_win_idx)}")

# ---------- export GRIB: (2) original + labels & time-score ----------
cluster_da_full = xr.DataArray(
    npx.broadcast_to(
        labels_time[:, None, None], (len(times), ds.sizes[lat_name], ds.sizes[lon_name])
    ),
    coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
    dims=("time", lat_name, lon_name),
    name="cluster_id",
    attrs={"long_name": "sequence cluster label (per time-step)", "units": "1"},
)
score_da_full = xr.DataArray(
    npx.broadcast_to(
        time_score[:, None, None], (len(times), ds.sizes[lat_name], ds.sizes[lon_name])
    ),
    coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
    dims=("time", lat_name, lon_name),
    name="cluster_score",
    attrs={"long_name": "majority-vote strength (0..1)", "units": "1"},
)
ds_with_label = ds.copy()
ds_with_label["cluster_id"] = cluster_da_full
ds_with_label["cluster_score"] = score_da_full
pd.DataFrame(
    {"time": times, "cluster_id": labels_time, "cluster_score": time_score}
).to_csv(OUT / "time_cluster_map.csv", index=False)
out_full_grib = OUT / "original_with_cluster_id.grib2"
export_grib_from_ds(ds_with_label, out_full_grib)
print(f"[DONE] Exported: {out_full_grib} (labels+scores) and per-window GRIBs")

# ---------- POST-PROCESS: probe point plots ----------
if (args.probe_lat is not None) and (args.probe_lon is not None) and (not args.no_plot):
    print("[INFO] Generating probe-point plots...")
    # extraire u10 & v10 au point (nearest) sur le dataset ORIGINAL (unités physiques)
    assert "u10" in ds and "v10" in ds, "u10 & v10 required in ds"
    u_pt = ds["u10"].sel(
        {lat_name: args.probe_lat, lon_name: args.probe_lon}, method="nearest"
    )
    v_pt = ds["v10"].sel(
        {lat_name: args.probe_lat, lon_name: args.probe_lon}, method="nearest"
    )
    plat = float(u_pt.coords[lat_name].values)
    plon = float(u_pt.coords[lon_name].values)
    print(f"[INFO] Probe nearest grid: lat={plat:.3f}, lon={plon:.3f}")

    # Séries physiques
    u_series = u_pt.values.astype(float)
    v_series = v_pt.values.astype(float)

    # Recompose toutes les fenêtres au point
    L = wins[0][1] - wins[0][0] + 1
    t_rel = npx.arange(L) * dt_hours
    seq_u = npx.array([u_series[s : e + 1] for (s, e) in wins])  # (N,L)
    seq_v = npx.array([v_series[s : e + 1] for (s, e) in wins])  # (N,L)

    # Membres par cluster & médonoïdes déjà calculés
    cluster_members = {c: npx.where(labels_seq == c)[0] for c in range(k)}

    # Figure : lignes = clusters, colonnes = 2 (u10, v10)
    nrows, ncols = k, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 2.6 * k), sharex=True)
    if k == 1:
        axes = npx.array([axes])
    axes = axes.reshape(k, 2)

    for c in range(k):
        members = cluster_members[c]
        if members.size == 0:
            for j in range(2):
                axes[c, j].set_title(f"Cluster {c} (empty)")
                axes[c, j].axis("off")
            continue

        # u10 panel
        ax_u = axes[c, 0]
        for m in members:
            ax_u.plot(t_rel, seq_u[m], color="0.7", lw=0.8, alpha=0.5)
        mu = seq_u[members].mean(axis=0)
        su = seq_u[members].std(axis=0)
        ax_u.fill_between(
            t_rel, mu - su, mu + su, alpha=0.15, linewidth=0, label="mean ± std"
        )
        ax_u.plot(t_rel, mu, lw=1.2, alpha=0.9, label="mean")
        # medoid
        m_win = medoid_win_idx.get(c, None)
        if m_win is not None:
            ax_u.plot(t_rel, seq_u[m_win], lw=2.3, alpha=0.95, label="medoid")
            s, e = wins[m_win]
            title_u = (
                f"Cluster {c} | u10 | medoid: {times[s].strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            title_u = f"Cluster {c} | u10"
        ax_u.set_title(title_u)
        ax_u.set_ylabel("u10 (m/s)")
        ax_u.grid(True, alpha=0.25)
        ax_u.legend(loc="upper right", fontsize=8)

        # v10 panel
        ax_v = axes[c, 1]
        for m in members:
            ax_v.plot(t_rel, seq_v[m], color="0.7", lw=0.8, alpha=0.5)
        mv = seq_v[members].mean(axis=0)
        sv = seq_v[members].std(axis=0)
        ax_v.fill_between(
            t_rel, mv - sv, mv + sv, alpha=0.15, linewidth=0, label="mean ± std"
        )
        ax_v.plot(t_rel, mv, lw=1.2, alpha=0.9, label="mean")
        if m_win is not None:
            ax_v.plot(t_rel, seq_v[m_win], lw=2.3, alpha=0.95, label="medoid")
            ax_v.set_title(f"Cluster {c} | v10 | medoid window")
        else:
            ax_v.set_title(f"Cluster {c} | v10")
        ax_v.set_ylabel("v10 (m/s)")
        ax_v.grid(True, alpha=0.25)
        ax_v.legend(loc="upper right", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Hours from window start")

    fig.suptitle(
        f"Probe point (lat={plat:.2f}, lon={plon:.2f})  |  windows={len(wins)}  |  k={k}",
        y=0.999,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png_path = OUT / args.plot_out
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"[PLOT] Saved probe plot → {png_path}")
