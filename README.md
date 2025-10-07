
# 🌊 Meteo-Scenarios  
**Reducing weather scenario sets for efficient and robust performance optimisation of wind-assisted ships**

---

## 🧭 Overview

**Meteo-Scenarios** is an open-source Python package developed for the study and optimisation of **Wind-Assisted Ships (WASP)** under realistic meteorological variability.  
It provides a complete and reproducible framework for **scenario reduction**, **time-series clustering**, and **synoptic-scale weather analysis**.

The objective is to reduce the computational burden associated with large-scale weather scenario sets, which often involve thousands of simulated voyages, while retaining the essential **statistical and dynamical variability** of the original data.

By identifying **representative synoptic sequences** using clustering techniques such as **MiniBatch K-Means** and **Dynamic Time Warping (DTW) + PAM medoids**, the method enables efficient yet robust performance prediction, design optimisation, and mission-level analysis of wind-assisted vessels.

---

## 🔬 Scientific Context

Performance prediction and optimisation of wind-assisted ships typically rely on **multi-year weather scenario ensembles**.  
Brute-force evaluation ensures statistical robustness but becomes **computationally prohibitive** when coupled with ship design optimisation, routing simulation, or fleet-level planning.

This package introduces a methodology for **synoptic scenario reduction** based on clustering of meteorological time-series windows.  
Each window represents a sequence of spatial weather states (e.g. 10 m wind, significant wave height), clustered according to similarity metrics tailored to time series:

- **Euclidean distance** for rapid analysis via MiniBatch K-Means  
- **Dynamic Time Warping (DTW)** for alignment-invariant temporal comparison, combined with **Partitioning Around Medoids (PAM)** for robust representative selection

The reduced scenario sets preserve both the **statistical diversity** and the **seasonality** of the original data, while reducing routing computations by **orders of magnitude**.

Such reduced sets can reproduce voyage-time and fuel-consumption distributions with high fidelity, making **joint design–routing optimisation** of WASP tractable in realistic industrial studies.

---

## ⚙️ Key Capabilities

- **Aggregation** of multiple GRIB/NetCDF datasets onto a unified grid and timeline  
- **Sliding-window sequence generation** and feature embedding (with optional PCA)  
- **Clustering** with K-Means or DTW+PAM medoids  
- **Export** of representative medoid scenarios as GRIB2 and CSV summaries  
- **Validation tools** for spatial maps and local time-series diagnostics  
- **Mission-level analysis** enabling integration of design, routing, and fleet-planning decisions under uncertainty

---

## 📚 Related Research

This package builds on methods widely used in other domains for time-series aggregation and uncertainty reduction:

- Paparrizos, J., *et al.* (2024). *Bridging Time Series Data Mining and Deep Learning via Representation Learning: A Survey.*  
- Teichgräber, H., *et al.* (2022). *Time-Series Aggregation for Energy System Design: Review and Application.*  
- Yerbury, E., *et al.* (2025). *Comparing Clustering Techniques for Renewable Energy Forecasting.*  
- Wei, L., *et al.* (2022). *Joint Design and Routing Optimization for Wind-Assisted Ships.*  
- Meng, Q., *et al.* (2014). *Containership Routing and Scheduling under Weather Uncertainty.*  
- Cao, J., *et al.* (2025). *Risk-Aware Fleet-Level Optimisation under Weather Uncertainty.*

---

## 🧩 Repository Structure

```bash
meteo-scenarios/
├─ src/meteo_scenarios/
│  ├─ io.py              # I/O utils (open, export GRIB, normalize dims)
│  ├─ gridtime.py        # Temporal & spatial alignment
│  ├─ clustering.py      # KMeans & DTW-PAM
│  ├─ windows.py         # Sliding window generation
│  ├─ export.py          # GRIB exports + label handling
│  ├─ plotting.py        # Maps & probe diagnostics
│  └─ cli/
│     ├─ aggregate.py    # meteo-aggregate
│     ├─ reduce.py       # meteo-reduce
│     ├─ plot_map.py     # meteo-plot-map
│     └─ probe.py        # meteo-probe
└─ examples/
   ├─ 01_merge_wind_wave.sh
   ├─ 02_reduce_sequences.sh
   └─ 03_probe_plots.sh

```

## 🚀 Typical Workflow

### 1 - Merge heterogeneous meteorological datasets
```bash
meteo-aggregate \
  --in "gribs/wind_2020.grib" \
  --in "gribs/wave_2020.grib" \
  --target-grid finer \
  --time-mode freq --time-freq 12H \
  --out merged_12H.grib2

```

### 2 - Reduce to representative synoptic scenarios
```bash
meteo-reduce merged_12H.grib2 \
  --vars u10,v10 \
  --window-hours 72 \
  --stride-hours 24 \
  --clusters 6 \
  --seq-metric euclid \
  --out reduced/
```

### 3 - Visualise and validate
```bash
meteo-plot-map reduced/original_with_cluster_id.grib2
meteo-probe reduced/original_with_cluster_id.grib2 \
  --probe-lat 45.0 --probe-lon -20.0 \
  --plot-out probe.png

```

## 🔧 Installation

```bash
git clone https://github.com/cdhainaut/meteo-clustering.git
cd meteo-clustering
pip install -e .

```
  
## 🧠 Research Scope

This tool supports studies on:

* Scenario reduction for performance prediction of WASP
* Design–routing coupling under uncertain meteorological forcing
* Fleet-level and mission-level optimisation
* Uncertainty quantification and robust decision-making for maritime decarbonisation

The resulting datasets and models can be used as reproducible inputs for optimisation frameworks, routing solvers, or Monte-Carlo performance assessment.

## 🧾 Citation
If you use this package in your research, please cite:

```bibtex
@inproceedings{dhainaut2025meteo,
  title   = {Reducing Weather Scenario Sets for Efficient and Robust Performance Optimisation of Wind-Assisted Ships},
  author  = {Dhainaut, Charles and Sacher, Mathieu},
  booktitle = {7th Innov'Sail Symposium},
  year    = {2026},
  address = {Göteborg, Sweden},
  organization = {ENSTA Bretagne, IRDL}
}
```
