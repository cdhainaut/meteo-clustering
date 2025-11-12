from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from .io import export_grib_from_ds


def majority_vote_labels(wins: List[Tuple[int, int]], labels_seq: np.ndarray, T: int):
    cover = [[] for _ in range(T)]
    for w_i, (s, e) in enumerate(wins):
        c = int(labels_seq[w_i])
        for t in range(s, e + 1):
            cover[t].append(c)
    labels_time = np.zeros(T, dtype=int)
    time_score = np.zeros(T, dtype=float)
    for t, lst in enumerate(cover):
        if not lst:
            labels_time[t] = -1
            time_score[t] = 0.0
        else:
            vals, counts = np.unique(np.array(lst), return_counts=True)
            imax = int(np.argmax(counts))
            labels_time[t] = int(vals[imax])
            time_score[t] = float(counts[imax]) / float(len(lst))
    return labels_time, time_score


def attach_time_labels_scores(
    ds: xr.Dataset,
    labels_time: np.ndarray,
    time_score: np.ndarray,
    lat_name="latitude",
    lon_name="longitude",
) -> xr.Dataset:
    T = ds.sizes["time"]
    lab = xr.DataArray(
        np.broadcast_to(
            labels_time[:, None, None], (T, ds.sizes[lat_name], ds.sizes[lon_name])
        ),
        coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_id",
        attrs={"long_name": "sequence cluster label (per time-step)", "units": "1"},
    )
    sc = xr.DataArray(
        np.broadcast_to(
            time_score[:, None, None], (T, ds.sizes[lat_name], ds.sizes[lon_name])
        ),
        coords={"time": ds["time"], lat_name: ds[lat_name], lon_name: ds[lon_name]},
        dims=("time", lat_name, lon_name),
        name="cluster_score",
        attrs={"long_name": "majority-vote strength (0..1)", "units": "1"},
    )
    out = ds.copy()
    out["cluster_id"] = lab
    out["cluster_score"] = sc
    return out


def export_medoids(
    ds: xr.Dataset,
    wins: List[Tuple[int, int]],
    medoid_win_idx: Dict[int, int],
    window_score: np.ndarray,
    outdir: Path,
    lat_name="latitude",
    lon_name="longitude",
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    times = pd.DatetimeIndex(ds["time"].values)
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
        cluster_vec = np.full(T, int(c), dtype=int)
        score_vec = np.full(T, float(window_score[w_i]), dtype=float)
        cluster_da = xr.DataArray(
            np.broadcast_to(
                cluster_vec[:, None, None],
                (T, sub.sizes[lat_name], sub.sizes[lon_name]),
            ),
            coords={
                "time": sub["time"],
                lat_name: sub[lat_name],
                lon_name: sub[lon_name],
            },
            dims=("time", lat_name, lon_name),
            name="cluster_id",
            attrs={"long_name": "sequence cluster label (window)", "units": "1"},
        )
        score_da = xr.DataArray(
            np.broadcast_to(
                score_vec[:, None, None], (T, sub.sizes[lat_name], sub.sizes[lon_name])
            ),
            coords={
                "time": sub["time"],
                lat_name: sub[lat_name],
                lon_name: sub[lon_name],
            },
            dims=("time", lat_name, lon_name),
            name="cluster_score",
            attrs={"long_name": "representativeness score (0..1)", "units": "1"},
        )
        sub_with = sub.copy()
        sub_with["cluster_id"] = cluster_da
        sub_with["cluster_score"] = score_da
        out_grib = outdir / f"scenario_seq_cluster{c}.grib2"
        if export_grib_from_ds(sub_with, out_grib):
            n_grib += 1
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "scenarios_sequences_summary.csv", index=False)
    print(f"[OK] Windows GRIB2 (with score): {n_grib}/{len(medoid_win_idx)}")
    return df
