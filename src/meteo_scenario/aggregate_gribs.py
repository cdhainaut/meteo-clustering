#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse, os, shutil, tempfile, glob
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Dict, Optional

# ----------------------- utils I/O -----------------------


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def export_grib_from_ds(ds: xr.Dataset, out_grib: Path) -> bool:
    tmpdir = Path(tempfile.mkdtemp(prefix="agg_any_"))
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
            print(f"[OK] GRIB2 → {out_grib}")
            return True
        print(f"[WARN] CDO failed ({ret}). Kept NetCDF fallback.")
    else:
        print("[WARN] 'cdo' not found. Keeping NetCDF and printing conversion hint.")
    fallback_nc = out_grib.with_suffix(".nc")
    ds.to_netcdf(fallback_nc)
    print(f"[OK] NetCDF → {fallback_nc}")
    print(f"→ Later: cdo -f grb2 copy '{fallback_nc}' '{out_grib}'")
    return False


# ----------------------- geo/time helpers -----------------------


def _rename_dims(ds: xr.Dataset) -> xr.Dataset:
    # time
    if "time" not in ds.dims:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        elif "forecast_time" in ds.coords:
            ds = ds.rename({"forecast_time": "time"})
        else:
            raise ValueError("No time dimension found.")
    # lat/lon
    lat = "latitude" if "latitude" in ds.dims else ("lat" if "lat" in ds.dims else None)
    lon = (
        "longitude" if "longitude" in ds.dims else ("lon" if "lon" in ds.dims else None)
    )
    if lat is None or lon is None:
        raise ValueError("Missing latitude/longitude dims.")
    ds = ds.rename({lat: "latitude", lon: "longitude"})
    return ds


def _ensure_monotonic_lat(ds: xr.Dataset) -> xr.Dataset:
    lat = ds["latitude"].values
    if lat[0] < lat[-1]:
        # many GRIBs are North-to-South; keep as-is (xarray interp works either way)
        return ds
    return ds  # both orders OK for xarray.interp


def _wrap_longitude_to(lon: np.ndarray, target: str) -> np.ndarray:
    if target == "negpos":  # [-180, 180]
        lon2 = ((lon + 180.0) % 360.0) - 180.0
    else:  # "pos360": [0, 360)
        lon2 = lon % 360.0
    # keep monotonic increasing if possible
    order = np.argsort(lon2)
    return lon2[order]


def _normalize_longitudes(ds: xr.Dataset, target: str) -> xr.Dataset:
    # detect current range
    lon = ds["longitude"].values
    if (lon.min() >= 0.0) and (lon.max() <= 360.0):
        src = "pos360"
    else:
        src = "negpos"
    if src == target:
        return ds.sortby(["longitude", "latitude", "time"])
    # convert coordinates (values only), then reindex
    lon_vals = ds["longitude"].values
    if target == "negpos":
        lon_new = ((lon_vals + 180.0) % 360.0) - 180.0
    else:
        lon_new = lon_vals % 360.0
    order = np.argsort(lon_new)
    ds = ds.assign_coords(longitude=(("longitude",), lon_new))
    ds = ds.isel(longitude=order)
    return ds.sortby(["longitude", "latitude", "time"])


def choose_target_grid(
    dsets: List[xr.Dataset], how: str, grid_file: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if grid_file:
        ref = xr.open_dataset(grid_file)
        ref = _rename_dims(ref)
        return ref["latitude"].values, ref["longitude"].values

    lats = [d["latitude"].values for d in dsets]
    lons = [d["longitude"].values for d in dsets]

    if how == "first":
        return lats[0], lons[0]
    if how == "finer":

        def res(arr):
            return np.min(np.abs(np.diff(arr))) if arr.size > 1 else np.inf

        dlat = min(res(a) for a in lats)
        dlon = min(res(a) for a in lons)
        lat_min = max(a.min() for a in lats)
        lat_max = min(a.max() for a in lats)
        lon_min = max(a.min() for a in lons)
        lon_max = min(a.max() for a in lons)
        if not (lat_min < lat_max and lon_min < lon_max):
            raise ValueError("No overlapping spatial extent across inputs.")
        # respect descending ERA5 latitude if majority is descending
        descending = sum(int(a[0] > a[-1]) for a in lats) > (len(lats) // 2)
        lat_t = np.arange(lat_min, lat_max + 1e-9, dlat)
        if descending:
            lat_t = lat_t[::-1]
        lon_t = np.arange(lon_min, lon_max + 1e-9, dlon)
        return lat_t, lon_t
    raise ValueError("Unknown target grid mode.")


def temporal_align_all(
    dsets: List[xr.Dataset], mode: str, freq: Optional[str]
) -> List[xr.Dataset]:
    if mode == "intersection":
        times = [pd.DatetimeIndex(ds["time"].values) for ds in dsets]
        common = times[0].values
        for t in times[1:]:
            common = np.intersect1d(common, t.values)
        if common.size == 0:
            raise ValueError("No common timestamps across inputs.")
        return [ds.sel(time=common) for ds in dsets]
    if mode == "freq":
        if not freq:
            raise ValueError("Provide --time-freq with --time-mode=freq")
        t0 = max(pd.DatetimeIndex(ds["time"].values).min() for ds in dsets)
        t1 = min(pd.DatetimeIndex(ds["time"].values).max() for ds in dsets)
        if t0 >= t1:
            raise ValueError("No overlapping time span across inputs.")
        target_t = pd.date_range(t0, t1, freq=freq)
        return [ds.interp(time=target_t) for ds in dsets]
    raise ValueError("Unknown temporal mode.")


# ----------------------- angle handling -----------------------


def is_angle_var(da: xr.DataArray, hint_names: set[str]) -> bool:
    name = da.name.lower() if da.name else ""
    if name in hint_names:
        return True
    units = str(da.attrs.get("units", "")).lower()
    if "degree" in units or "deg" in units:
        # heuristique : variables 0..360 ou -180..180
        v = da.values
        try:
            finite = np.asarray(v[np.isfinite(v)])
            if finite.size and (finite.min() >= -360) and (finite.max() <= 360):
                return True
        except Exception:
            pass
    return False


def circular_mean_deg(angles_deg: xr.DataArray, dim="time") -> xr.DataArray:
    ang = np.deg2rad(angles_deg)
    s, c = np.sin(ang), np.cos(ang)
    m = np.arctan2(s.mean(dim=dim), c.mean(dim=dim))
    return (np.rad2deg(m) + 360.0) % 360.0


# ----------------------- main -----------------------


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate N GRIB/NetCDF onto a common grid/time and export a single GRIB2."
    )
    ap.add_argument(
        "--in",
        dest="inputs",
        action="append",
        required=True,
        help="Input files or glob patterns. Can be repeated.",
    )
    ap.add_argument(
        "--target-grid",
        choices=["first", "finer"],
        default="finer",
        help="Spatial target grid: use first dataset grid or the finest intersection.",
    )
    ap.add_argument(
        "--target-grid-file",
        type=str,
        default=None,
        help="Optional reference file providing target (lat,lon). Overrides --target-grid.",
    )
    ap.add_argument(
        "--time-mode",
        choices=["intersection", "freq"],
        default="intersection",
        help="Temporal alignment: intersection of timestamps or regular frequency.",
    )
    ap.add_argument(
        "--time-freq",
        type=str,
        default=None,
        help="e.g. '6H','12H','1D' (used with --time-mode=freq).",
    )
    ap.add_argument(
        "--resample-average",
        action="store_true",
        help="If set with --time-mode=freq, average on the target step instead of pure interpolation. "
        "Angles handled by circular mean.",
    )
    ap.add_argument(
        "--lon-range",
        choices=["negpos", "pos360"],
        default="negpos",
        help="Normalize longitudes to [-180,180] or [0,360).",
    )
    ap.add_argument(
        "--angle-vars",
        type=str,
        default="mwd,wind_direction,dir",
        help="Comma-separated list of variable names to treat as angles in degrees.",
    )
    ap.add_argument(
        "--prefix-collisions",
        action="store_true",
        help="Prefix variable names with file stem if name collisions occur.",
    )
    ap.add_argument("--out", type=str, default="merged.grib2")
    args = ap.parse_args()

    # expand globs
    files: List[str] = []
    for item in args.inputs:
        files.extend(glob.glob(item))
    files = [str(Path(f)) for f in files]
    if not files:
        raise SystemExit("No input files matched.")
    print(f"[INFO] Inputs ({len(files)}):")
    for f in files:
        print("  -", f)

    # load / normalize
    dsets: List[Tuple[str, xr.Dataset]] = []
    for f in files:
        ds = xr.open_dataset(f)  # cfgrib or netcdf engine auto
        ds = _rename_dims(ds)
        ds = _ensure_monotonic_lat(ds)
        ds = _normalize_longitudes(ds, target=args.lon_range)
        ds = ds.sortby(["latitude", "longitude", "time"])
        dsets.append((f, ds))

    # temporal alignment
    aligned = temporal_align_all(
        [ds for _, ds in dsets], mode=args.time_mode, freq=args.time_freq
    )

    # choose spatial grid
    lat_t, lon_t = choose_target_grid(
        aligned, how=args.target_grid, grid_file=args.target_grid_file
    )

    # spatial interp to target grid
    interp_all: List[Tuple[str, xr.Dataset]] = []
    for fname, ds in zip(files, aligned):
        ds_i = ds.interp(latitude=lat_t, longitude=lon_t)
        interp_all.append((fname, ds_i))

    # optional temporal average (when using regular freq)
    if args.time_mode == "freq" and args.resample_average and args.time_freq:
        # we resample each dataset; for angle vars use circular mean
        hint = set(v.strip().lower() for v in args.angle_vars.split(",") if v.strip())
        averaged: List[Tuple[str, xr.Dataset]] = []
        for fname, ds in interp_all:
            numeric_vars = []
            angle_vars = []
            for v in ds.data_vars:
                da = ds[v]
                if is_angle_var(da, hint):
                    angle_vars.append(v)
                else:
                    numeric_vars.append(v)
            parts = []
            if numeric_vars:
                parts.append(
                    ds[numeric_vars].resample(time=args.time_freq).mean(keep_attrs=True)
                )
            if angle_vars:
                # circular mean per angle var
                circ = []
                for v in angle_vars:
                    m = (
                        ds[v]
                        .resample(time=args.time_freq)
                        .map(lambda x: circular_mean_deg(x, dim="time"))
                    )
                    circ.append(m.rename(v))
                parts.append(xr.merge(circ))
            ds_avg = xr.merge(parts).sortby(["latitude", "longitude", "time"])
            averaged.append((fname, ds_avg))
        interp_all = averaged

    # merge all variables; handle name collisions if requested
    merged_vars: Dict[str, xr.DataArray] = {}
    seen: set[str] = set()
    for fname, ds in interp_all:
        stem = Path(fname).stem
        for v in ds.data_vars:
            name = v
            if (name in seen) and args.prefix_collisions:
                name = f"{stem}__{v}"
            if name in seen:
                # last resort: numeric suffix
                k = 2
                while f"{name}_{k}" in seen:
                    k += 1
                name = f"{name}_{k}"
            da = ds[v]
            da = da.rename(name)
            merged_vars[name] = da
            seen.add(name)

    merged = xr.Dataset(
        merged_vars,
        coords=dict(
            time=interp_all[0][1]["time"],
            latitude=("latitude", lat_t),
            longitude=("longitude", lon_t),
        ),
    )
    merged = merged.sortby(["latitude", "longitude", "time"])
    merged.attrs.update(
        dict(
            description="Merged multi-file product (any variables) on common grid/time.",
            note="Angles averaged with circular mean when --resample-average is set.",
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_grib_from_ds(merged, out)


if __name__ == "__main__":
    main()
