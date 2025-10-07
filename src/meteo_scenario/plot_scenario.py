#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nomad.numpy as np
import numpy as npx
import seaborn as sns
import cartopy.feature as cfeature
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation

sns.set_theme()
plt.rcParams["savefig.dpi"] = 150

parser = argparse.ArgumentParser()
parser.add_argument(
    "grib", type=str, help="Path to GRIB/NetCDF (with cluster_id/score if available)"
)
parser.add_argument(
    "--coarsen", type=int, default=None, help="Spatial coarsen factor (1=off)"
)
parser.add_argument(
    "--labels-csv",
    type=str,
    default=None,
    help="Optional CSV with columns time,cluster_id,cluster_score (fallback if not in GRIB)",
)
args = parser.parse_args()

# -------- Load ----------
ds = xr.open_dataset(args.grib)
ds = ds.sortby(["latitude", "longitude", "time"])

if args.coarsen is not None:
    ds = ds.coarsen(
        latitude=args.coarsen, longitude=args.coarsen, boundary="trim"
    ).mean()

# -------- Try to get cluster labels & scores ----------
lat_name = "latitude" if "latitude" in ds.dims else "lat"
lon_name = "longitude" if "longitude" in ds.dims else "lon"

cluster_per_t = None
score_per_t = None

if "cluster_id" in ds:
    try:
        cluster_per_t = (
            ds["cluster_id"].isel({lat_name: 0, lon_name: 0}).values.astype(int)
        )
    except Exception:
        cluster_per_t = ds["cluster_id"].values.astype(int)
if "cluster_score" in ds:
    try:
        score_per_t = (
            ds["cluster_score"].isel({lat_name: 0, lon_name: 0}).values.astype(float)
        )
    except Exception:
        score_per_t = ds["cluster_score"].values.astype(float)

# CSV fallback
if (cluster_per_t is None or score_per_t is None) and args.labels_csv is not None:
    lab = pd.read_csv(args.labels_csv)
    lab["time"] = pd.to_datetime(lab["time"]).values.astype("datetime64[ns]")
    times_dt = ds["time"].to_pandas().astype("datetime64[ns]")
    merged = pd.DataFrame({"time": times_dt}).merge(lab, on="time", how="left")
    if cluster_per_t is None:
        cluster_per_t = merged["cluster_id"].fillna(-1).astype(int).to_numpy()
    if score_per_t is None:
        if "cluster_score" in merged:
            score_per_t = merged["cluster_score"].fillna(0.0).astype(float).to_numpy()
        else:
            score_per_t = npx.zeros(len(times_dt), dtype=float)

# -------- Prepare arrays ----------
ds["time_float"] = ds["time"].astype(np.float64)
longitudes = ds["longitude"].to_numpy()
latitudes = ds["latitude"].to_numpy()
times_float = ds["time_float"].to_numpy()
u10 = ds["u10"].to_numpy()
v10 = ds["v10"].to_numpy()

tws = np.sqrt(u10**2 + v10**2)  # m/s
time_grid, lat_grid, lon_grid = np.meshgrid(
    times_float, latitudes, longitudes, indexing="ij"
)
n_steps = u10.shape[0]
times_dt = ds["time"].to_numpy()

# -------- Plot ----------
fig = plt.figure(figsize=(8, 4))
projection = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=projection)
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])

ax.set_extent(
    [
        float(longitudes.min()),
        float(longitudes.max()),
        float(latitudes.min()),
        float(latitudes.max()),
    ],
    crs=projection,
)
ax.add_feature(
    cfeature.COASTLINE.with_scale("110m"), linewidth=0.7, edgecolor="black", zorder=6
)

contour = ax.contourf(
    longitudes,
    latitudes,
    tws[0] * 1.94384,
    cmap="Spectral_r",
    levels=200,
    vmin=0,
    transform=projection,
)
cb = fig.colorbar(contour, cax=cax)

quiv = ax.quiver(
    lon_grid[0],
    lat_grid[0],
    u10[0],
    v10[0],
    pivot="tip",
    color="white",
    alpha=0.8,
    scale=5e2,
    transform=projection,
    zorder=5,
)


def fmt_title(i):
    ttxt = times_dt[i].astype("M8[s]").astype(str)
    if cluster_per_t is not None and score_per_t is not None:
        return (
            f"{ttxt}  |  cluster={int(cluster_per_t[i])}  |  score={score_per_t[i]:.2f}"
        )
    elif cluster_per_t is not None:
        return f"{ttxt}  |  cluster={int(cluster_per_t[i])}"
    else:
        return ttxt


title = ax.set_title(fmt_title(0), fontsize=12)

box_text = None
if cluster_per_t is not None:
    s_txt = f"cluster: {int(cluster_per_t[0])}"
    if score_per_t is not None:
        s_txt += f"\nscore: {score_per_t[0]:.2f}"
    box_text = ax.text(
        0.01,
        0.98,
        s_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
        zorder=7,
    )


def update(i):
    global contour
    for c in contour.collections:
        c.remove()
    contour = ax.contourf(
        longitudes,
        latitudes,
        tws[i] * 1.94384,
        cmap="Spectral_r",
        levels=200,
        vmin=0,
        transform=projection,
        zorder=2,
    )
    cb.update_normal(contour)
    quiv.set_UVC(u10[i], v10[i])
    title.set_text(fmt_title(i))
    if box_text is not None:
        s_txt = f"cluster: {int(cluster_per_t[i])}"
        if score_per_t is not None:
            s_txt += f"\nscore: {score_per_t[i]:.2f}"
        box_text.set_text(s_txt)
    return (
        [contour, quiv, title, box_text]
        if box_text is not None
        else [contour, quiv, title]
    )


ani = FuncAnimation(fig, update, frames=n_steps, interval=800, blit=False)
plt.show()
