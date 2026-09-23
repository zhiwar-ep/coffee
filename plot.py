import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DEFAULT_STATS_FILE = "postProcessing/solidStats/0/solidStats.dat"

# Optional:
# python3 plot.py path/to/solidStats.dat
STATS_FILE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STATS_FILE

OUTPUT_POROSITY_VOLUME = "porosity_relative_volume.pdf"
OUTPUT_TIME_EVOLUTION = "volume_porosity_time.pdf"
OUTPUT_FIT = "fit.txt"

# Data rows to use (1-based, inclusive)
ROW_START = 1
ROW_END = None


# ============================================================
# READ solidStats.dat
# ============================================================
#
# Expected columns:
#
# 0  Time
# 1  CellsBelowCriticalPorosity
# 2  RegionVolume
# 3  TotalYrawAll
# 4  TotalYrawBelowCriticalPorosity
# 5  AvgPorosityBelowCriticalPorosity
# 6  AvgPorosityInitialRegion
#
# Used here:
#
# Time
# RegionVolume
# AvgPorosityBelowCriticalPorosity
#
# ============================================================

stats_data = np.loadtxt(
    STATS_FILE,
    comments="#"
)

stats_data = np.atleast_2d(
    stats_data
)

if stats_data.shape[1] < 7:
    raise ValueError(
        f"{STATS_FILE} must contain at least 7 columns."
    )


time_all = stats_data[:, 0]
region_volume_all = stats_data[:, 2]
avg_porosity_all = stats_data[:, 5]


# ============================================================
# SELECT DATA ROWS
# ============================================================

n_rows = len(time_all)

if ROW_START is None:
    ROW_START = 1

if ROW_START < 1 or ROW_START > n_rows:
    raise ValueError(
        f"ROW_START must be between 1 and {n_rows}."
    )

if ROW_END is None:
    row_end_actual = n_rows
else:
    row_end_actual = ROW_END

if (
    row_end_actual < ROW_START
    or row_end_actual > n_rows
):
    raise ValueError(
        f"ROW_END must be between ROW_START ({ROW_START}) "
        f"and the last row ({n_rows})."
    )


row_slice = slice(
    ROW_START - 1,
    row_end_actual
)


time = time_all[row_slice]
region_volume = region_volume_all[row_slice]
avg_porosity = avg_porosity_all[row_slice]


# ============================================================
# INITIAL VOLUME
# ============================================================

V0 = region_volume_all[0]

if V0 == 0:
    raise ValueError(
        "Initial RegionVolume is zero, "
        "so V/V0 cannot be calculated."
    )


# ============================================================
# RELATIVE VOLUME
# ============================================================

relative_volume = region_volume / V0


# ============================================================
# VALID DATA FOR POROSITY - VOLUME ANALYSIS
# ============================================================

valid = (
    np.isfinite(relative_volume)
    & np.isfinite(avg_porosity)
)

volume_plot = relative_volume[valid]
porosity_plot = avg_porosity[valid]


if len(volume_plot) < 2:
    raise ValueError(
        "At least two valid points are required "
        "for linear fitting."
    )


# ============================================================
# LINEAR FIT
#
# Porosity = slope * (V/V0) + intercept
# ============================================================

slope, intercept = np.polyfit(
    volume_plot,
    porosity_plot,
    1
)


# Predicted porosity at simulation points
porosity_predicted = (
    slope * volume_plot
    + intercept
)


# ============================================================
# FIT QUALITY
# ============================================================

residuals = (
    porosity_plot
    - porosity_predicted
)

ss_res = np.sum(
    residuals**2
)

ss_tot = np.sum(
    (
        porosity_plot
        - np.mean(porosity_plot)
    )**2
)

if ss_tot > 0:
    r_squared = (
        1.0
        - ss_res / ss_tot
    )
else:
    r_squared = np.nan


rmse = np.sqrt(
    np.mean(
        residuals**2
    )
)


# ============================================================
# CREATE FIT LINE
# ============================================================

volume_fit = np.linspace(
    volume_plot.min(),
    volume_plot.max(),
    300
)

porosity_fit = (
    slope * volume_fit
    + intercept
)


# ============================================================
# WRITE FIT INFORMATION
# ============================================================

with open(
    OUTPUT_FIT,
    "w"
) as fit_file:

    fit_file.write(
        "============================================================\n"
    )

    fit_file.write(
        "LINEAR FIT: POROSITY VS RELATIVE VOLUME\n"
    )

    fit_file.write(
        "============================================================\n\n"
    )

    fit_file.write(
        f"Input file = {STATS_FILE}\n"
    )

    fit_file.write(
        f"Rows used = {ROW_START} to {row_end_actual}\n"
    )

    fit_file.write(
        f"Number of data points = {len(volume_plot)}\n"
    )

    fit_file.write(
        f"Initial volume V0 = {V0:.10e}\n\n"
    )

    fit_file.write(
        "Model:\n"
    )

    fit_file.write(
        "Porosity = slope * (V/V0) + intercept\n\n"
    )

    fit_file.write(
        f"Slope     = {slope:.10f}\n"
    )

    fit_file.write(
        f"Intercept = {intercept:.10f}\n"
    )

    fit_file.write(
        f"R_squared = {r_squared:.10f}\n"
    )

    fit_file.write(
        f"RMSE      = {rmse:.10f}\n\n"
    )

    fit_file.write(
        "Equation:\n"
    )

    fit_file.write(
        f"Porosity = {slope:.10f} * (V/V0) "
        f"+ ({intercept:.10f})\n"
    )


print(
    "============================================================"
)

print(
    "LINEAR FIT"
)

print(
    "============================================================"
)

print(
    f"Porosity = {slope:.6f} * (V/V0) "
    f"+ ({intercept:.6f})"
)

print(
    f"R² = {r_squared:.6f}"
)

print(
    f"RMSE = {rmse:.6f}"
)

print(
    f"Fit information saved to: {OUTPUT_FIT}"
)


# ============================================================
# PLOT 1
# POROSITY VS RELATIVE VOLUME + LINEAR FIT
# ============================================================

plt.figure(
    figsize=(6, 5)
)


# Simulation data
plt.plot(
    volume_plot,
    porosity_plot,
    linewidth=1.5,
    marker="o",
    markersize=3,
    label="Simulation"
)


# Linear fit
plt.plot(
    volume_fit,
    porosity_fit,
    linestyle="--",
    linewidth=2.0,
    label=(
        f"Linear fit "
        f"(R² = {r_squared:.3f})"
    )
)


plt.xlabel(
    "Relative Volume (V/V0)"
)

plt.ylabel(
    "Porosity"
)

plt.title(
    "Porosity vs Relative Volume"
)


plt.grid(
    True,
    linestyle=":",
    linewidth=0.7
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_POROSITY_VOLUME,
    bbox_inches="tight"
)


print(
    f"Plot saved to: "
    f"{OUTPUT_POROSITY_VOLUME}"
)


# ============================================================
# PLOT 2
# VOLUME AND POROSITY VS TIME
# ============================================================

valid_time = (
    np.isfinite(time)
    & np.isfinite(relative_volume)
    & np.isfinite(avg_porosity)
)


time_plot = time[valid_time]

volume_time_plot = (
    relative_volume[valid_time]
)

porosity_time_plot = (
    avg_porosity[valid_time]
)


plt.figure(
    figsize=(7, 5)
)


plt.plot(
    time_plot,
    volume_time_plot,
    linewidth=2.0,
    label="Relative volume (V/V0)"
)


plt.plot(
    time_plot,
    porosity_time_plot,
    linewidth=2.0,
    label="Porosity"
)


plt.xlabel(
    "Time"
)

plt.ylabel(
    "Value"
)

plt.title(
    "Volume and Porosity Evolution"
)


plt.grid(
    True,
    linestyle=":",
    linewidth=0.7
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_TIME_EVOLUTION,
    bbox_inches="tight"
)


print(
    f"Plot saved to: "
    f"{OUTPUT_TIME_EVOLUTION}"
)


plt.show()