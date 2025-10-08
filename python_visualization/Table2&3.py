import numpy as np
import rasterio
import glob
import os
import re
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, balanced_accuracy_score, precision_score
import pandas as pd

# ======================
# Set input/output paths
# ======================
srbdf_folder = r"data:\meltarea\SRBDF"
ascat_folder = r"data:\meltarea\ASCAT"
sentinel_folder = r"data:\meltarea\Sentinel-1"
output_folder = r"data:\Accuracy_Result"
os.makedirs(output_folder, exist_ok=True)

# ======================
# Helper functions
# ======================
def extract_date(filename):
    """Extract an 8-digit date string (YYYYMMDD) from filename."""
    match = re.search(r'\d{8}', filename)
    return match.group() if match else None

def build_file_map(folder):
    """Build a mapping from date string to file path for .tif files."""
    files = glob.glob(os.path.join(folder, '*.tif'))
    return {extract_date(os.path.basename(f)): f for f in files if extract_date(os.path.basename(f))}

# Build maps
srbdf_map = build_file_map(srbdf_folder)
ascat_map = build_file_map(ascat_folder)
sentinel_map = build_file_map(sentinel_folder)

# Find common dates
srbdf_ascat_dates = sorted(set(srbdf_map) & set(ascat_map))
srbdf_sentinel_dates = sorted(set(srbdf_map) & set(sentinel_map))

# ======================
# Pixel-level overall metrics
# ======================
def compute_overall_metrics(date_list, pred_map, ref_map, label):
    """Compute pixel-wise metrics by aggregating all available images."""
    y_pred_all, y_true_all = [], []

    for date in date_list:
        pred_path = pred_map[date]
        ref_path = ref_map[date]
        with rasterio.open(pred_path) as pred_ds, rasterio.open(ref_path) as ref_ds:
            pred = pred_ds.read(1)
            ref = ref_ds.read(1)
            if pred.shape != ref.shape:
                continue

            pred_nodata = pred_ds.nodata
            ref_nodata = ref_ds.nodata
            mask = (
                (pred != pred_nodata if pred_nodata is not None else True) &
                (ref != ref_nodata if ref_nodata is not None else True) &
                np.isin(pred, [0, 1]) & np.isin(ref, [0, 1])
            )
            if not np.any(mask):
                continue

            y_pred_all.extend(pred[mask].astype(int).flatten())
            y_true_all.extend(ref[mask].astype(int).flatten())

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2 or len(np.unique(y_pred_all)) < 2:
        return {"Comparison": label, "OA": np.nan, "Kappa": np.nan, "F1": np.nan, "Precision": np.nan, "BalAcc": np.nan}

    return {
        "Comparison": label,
        "OA": accuracy_score(y_true_all, y_pred_all),
        "Kappa": cohen_kappa_score(y_true_all, y_pred_all),
        "F1": f1_score(y_true_all, y_pred_all),
        "Precision": precision_score(y_true_all, y_pred_all),
        "BalAcc": balanced_accuracy_score(y_true_all, y_pred_all)
    }

overall_pixel_metrics = []
overall_pixel_metrics.append(compute_overall_metrics(srbdf_ascat_dates, srbdf_map, ascat_map, "SRBDF_vs_ASCAT"))
overall_pixel_metrics.append(compute_overall_metrics(srbdf_sentinel_dates, srbdf_map, sentinel_map, "SRBDF_vs_Sentinel"))

df_overall_pixel = pd.DataFrame(overall_pixel_metrics)
df_overall_pixel.to_csv(os.path.join(output_folder, "Overall_Pixelwise_Validation.csv"), index=False)
print("Overall pixel-wise validation results:")
print(df_overall_pixel)

# ======================
# Seasonal pixel-level metrics
# ======================
def get_winter_season(date):
    """Nov–Dec → year–year+1; Jan–Mar → year-1–year"""
    year = date.year
    month = date.month
    if month >= 11:
        return f"{year}-{year+1}"
    elif month <= 3:
        return f"{year-1}-{year}"
    else:
        return None

def compute_seasonal_pixel_metrics(date_list, pred_map, ref_map, label):
    """Compute seasonal pixel-level metrics."""
    seasonal_pixel_metrics = {}
    for date in date_list:
        season = get_winter_season(pd.to_datetime(date))
        if season is None:
            continue

        if season not in seasonal_pixel_metrics:
            seasonal_pixel_metrics[season] = {"y_pred": [], "y_true": []}

        pred_path = pred_map[date]
        ref_path = ref_map[date]
        with rasterio.open(pred_path) as pred_ds, rasterio.open(ref_path) as ref_ds:
            pred = pred_ds.read(1)
            ref = ref_ds.read(1)
            if pred.shape != ref.shape:
                continue

            pred_nodata = pred_ds.nodata
            ref_nodata = ref_ds.nodata
            mask = (
                (pred != pred_nodata if pred_nodata is not None else True) &
                (ref != ref_nodata if ref_nodata is not None else True) &
                np.isin(pred, [0, 1]) & np.isin(ref, [0, 1])
            )
            if not np.any(mask):
                continue

            seasonal_pixel_metrics[season]["y_pred"].extend(pred[mask].astype(int).flatten())
            seasonal_pixel_metrics[season]["y_true"].extend(ref[mask].astype(int).flatten())

    results = []
    for season, data in seasonal_pixel_metrics.items():
        y_pred_all = np.array(data["y_pred"])
        y_true_all = np.array(data["y_true"])

        if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2 or len(np.unique(y_pred_all)) < 2:
            metrics = {"Winter_Season": season, "Comparison": label,
                       "OA": np.nan, "Kappa": np.nan, "F1": np.nan,
                       "Precision": np.nan, "BalAcc": np.nan}
        else:
            metrics = {
                "Winter_Season": season,
                "Comparison": label,
                "OA": accuracy_score(y_true_all, y_pred_all),
                "Kappa": cohen_kappa_score(y_true_all, y_pred_all),
                "F1": f1_score(y_true_all, y_pred_all),
                "Precision": precision_score(y_true_all, y_pred_all),
                "BalAcc": balanced_accuracy_score(y_true_all, y_pred_all)
            }
        results.append(metrics)

    return results

seasonal_pixel_metrics = []
seasonal_pixel_metrics.extend(compute_seasonal_pixel_metrics(srbdf_ascat_dates, srbdf_map, ascat_map, "SRBDF_vs_ASCAT"))
seasonal_pixel_metrics.extend(compute_seasonal_pixel_metrics(srbdf_sentinel_dates, srbdf_map, sentinel_map, "SRBDF_vs_Sentinel"))

df_seasonal_pixel = pd.DataFrame(seasonal_pixel_metrics)
df_seasonal_pixel.to_csv(os.path.join(output_folder, "Seasonal_Pixelwise_Validation.csv"), index=False)
print("Seasonal pixel-wise validation results:")
print(df_seasonal_pixel)
