import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nomad.numpy as np
import seaborn as sns
import cartopy.feature as cfeature

import xarray as xr

sns.set_theme()
import cartopy.crs as ccrs

time_range = ("2020-01-01", "2020-12-30")  # 15 jours seulement
main_grib = "wind_2020.grib"
ds = xr.open_dataset(main_grib)
