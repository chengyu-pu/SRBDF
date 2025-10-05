import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_squared_error
import pandas as pd

# ===========================
# Configuration and Settings
# ===========================

# Set plotting font (for figures in the manuscript)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Input directories containing daily melt area binary masks (1 = melt, 0 = no melt)
srbdf_folder = "data/SRBDF"
ascat_folder = "data/ASCAT"
sentinel_folder = "data/Sentinel-1"

# Area represented by one pixel (in km²)
pixel_area = 0.04  

# ===========================
# Collect all available files
# ===========================
srbdf_files = sorted(glob.glob(os.path.join(srbdf_folder, "*.tif")))
ascat_files = sorted(glob.glob(os.path.join(ascat_folder, "*.tif")))
sentinel_files = sorted(glob.glob(os.path.join(sentinel_folder, "*.tif")))

# Extract dates from filenames (assuming filenames are formatted as YYYYMMDD.tif)
srbdf_dates = [os.path.basename(f).split('.')[0] for f in srbdf_files]
ascat_dates = [os.path.basename(f).split('.')[0] for f in ascat_files]
sentinel_dates = [os.path.basename(f).split('.')[0] for f in sentinel_files]

# Generate a union set of all dates across three datasets
all_dates = sorted(set(srbdf_dates) | set(ascat_dates) | set(sentinel_dates))

# ===========================
# Define function to compute melt area
# ===========================
def compute_melt_area(file, pixel_size):
    """Compute total melt area (km²) from a binary raster file."""
    with rasterio.open(file) as src:
        data = src.read(1)
        return np.sum(data == 1) * pixel_size

# Initialize dataframe to store melt area from three datasets
df = pd.DataFrame(index=all_dates, columns=["SRBDF", "ASCAT", "Sentinel-1"])

# Fill dataframe with computed melt area for each available date
for date in all_dates:
    srbdf_file = os.path.join(srbdf_folder, f"{date}.tif")
    ascat_file = os.path.join(ascat_folder, f"{date}.tif")
    sentinel_file = os.path.join(sentinel_folder, f"{date}.tif")
    
    df.loc[date, "SRBDF"] = compute_melt_area(srbdf_file, pixel_area) if os.path.exists(srbdf_file) else np.nan
    df.loc[date, "ASCAT"] = compute_melt_area(ascat_file, pixel_area) if os.path.exists(ascat_file) else np.nan
    df.loc[date, "Sentinel-1"] = compute_melt_area(sentinel_file, pixel_area) if os.path.exists(sentinel_file) else np.nan

# Ensure numerical type
df = df.astype(float)

# ===========================
# Define metrics: correlation, bias, RMSE
# ===========================
def compute_metrics(pred, true):
    """Compute correlation (R), bias, and RMSE between two series."""
    valid_mask = ~np.isnan(pred) & ~np.isnan(true)
    if valid_mask.sum() > 0:
        R, _ = pearsonr(pred[valid_mask], true[valid_mask])
        bias = np.mean(pred[valid_mask] - true[valid_mask])
        rmse = np.sqrt(mean_squared_error(true[valid_mask], pred[valid_mask]))
        return R, bias, rmse
    else:
        return np.nan, np.nan, np.nan

# Compute statistics for SRBDF vs ASCAT and SRBDF vs Sentinel-1
R_srbdf_ascat, _, _ = compute_metrics(df["ASCAT"], df["SRBDF"])
R_srbdf_sentinel, _, _ = compute_metrics(df["Sentinel-1"], df["SRBDF"])

# ===========================
# Linear regression fitting
# ===========================
def fit_line(x, y):
    """Fit linear regression line y = slope * x + intercept."""
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, _, _, _ = linregress(x[valid_mask], y[valid_mask])
    return slope, intercept

slope_ascat, intercept_ascat = fit_line(df["ASCAT"], df["SRBDF"])
slope_sentinel, intercept_sentinel = fit_line(df["Sentinel-1"], df["SRBDF"])

def format_fit_equation(slope, intercept):
    """Format regression equation for figure legend."""
    if intercept >= 0:
        return f"y = {slope:.2f}x + {intercept:.2f}"
    else:
        return f"y = {slope:.2f}x - {abs(intercept):.2f}"

# ===========================
# Visualization
# ===========================

sns.set(style="whitegrid", font_scale=1.2)
empty_line = plt.Line2D([], [], linestyle='')  # placeholder for legend text

# --- Scatterplot: ASCAT vs SRBDF ---
fig1, ax1 = plt.subplots(figsize=(6, 5))
sns.scatterplot(x="ASCAT", y="SRBDF", data=df, ax=ax1, color='royalblue', s=50, legend=False)

# Regression fit line and 1:1 reference line
x_vals = np.array(ax1.get_xlim())
y_fit = slope_ascat * x_vals + intercept_ascat
line_fit_ascat, = ax1.plot(x_vals, y_fit, 'r--')  
line_yx_ascat, = ax1.plot(x_vals, x_vals, 'k-')  

# Axis labels and aspect ratio
ax1.set_xlabel("ASCAT Melt Area (km²)")
ax1.set_ylabel("SRBDF Melt Area (km²)")
ax1.set_aspect('equal', adjustable='box')

# Add legend with regression equation and R²
r_squared_ascat = R_srbdf_ascat**2
legend_labels_ascat = [
    f'Fit: {format_fit_equation(slope_ascat, intercept_ascat)}',
    'y = x',
    f'R² = {r_squared_ascat:.2f}'
]
ax1.legend(handles=[line_fit_ascat, line_yx_ascat, empty_line], labels=legend_labels_ascat, loc='upper left')

plt.tight_layout()
plt.show()

# --- Scatterplot: Sentinel-1 vs SRBDF ---
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.scatterplot(x="Sentinel-1", y="SRBDF", data=df, ax=ax2, color='royalblue', s=50, legend=False)

# Regression fit line and 1:1 reference line
x_vals2 = np.array(ax2.get_xlim())
y_fit2 = slope_sentinel * x_vals2 + intercept_sentinel
line_fit_sentinel, = ax2.plot(x_vals2, y_fit2, 'r--')  
line_yx_sentinel, = ax2.plot(x_vals2, x_vals2, 'k-')  

ax2.set_xlabel("Sentinel-1 Melt Area (km²)")
ax2.set_ylabel("SRBDF Melt Area (km²)")
ax2.set_aspect('equal', adjustable='box')

# Add legend with regression equation and R²
r_squared_sentinel = R_srbdf_sentinel**2
legend_labels_sentinel = [
    f'Fit: {format_fit_equation(slope_sentinel, intercept_sentinel)}',
    'y = x',
    f'R² = {r_squared_sentinel:.2f}'
]
ax2.legend(handles=[line_fit_sentinel, line_yx_sentinel, empty_line], labels=legend_labels_sentinel, loc='upper left')

plt.tight_layout()
plt.show()
