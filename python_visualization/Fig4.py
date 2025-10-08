import os
import re
from datetime import datetime, timedelta
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import pearsonr
import matplotlib.lines as mlines
import csv
from dateutil import parser

# Define the station coordinates (longitude, latitude)
station_lon = -65.2
station_lat = -67.75

# Input folders
srbdf_folder = r"data:\SRBDF"
ascat_folder = r"data:\ASCAT"    
sentinel_folder = r"data:\Sentinel-1"

# Load one SRBDF file to get CRS projection
srbdf_files = [f for f in os.listdir(srbdf_folder) if f.endswith(".tif")]
sample_tif = os.path.join(srbdf_folder, srbdf_files[0])
print("Reference raster file used:", sample_tif)

with rasterio.open(sample_tif) as src:
    crs_proj = src.crs

# Define transformers between geographic (EPSG:4326) and projection CRS
transformer_to_proj = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
transformer_to_geo = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)

# Convert the station coordinates into the target projection
x_proj, y_proj = transformer_to_proj.transform(station_lon, station_lat)
target_point = (x_proj, y_proj)

print("Projected station coordinates:", target_point)
print("Geographic coordinates (lon, lat):", (station_lon, station_lat))

# Function to extract pixel value from raster at given coordinates
def extract_pixel_value(tif_path, x, y):
    with rasterio.open(tif_path) as src:
        try:
            row, col = src.index(x, y)
            value = src.read(1)[row, col]
            if src.nodata is not None and value == src.nodata:
                return None
            return value
        except:
            return None

# Function to parse acquisition date from filename
def parse_date_from_filename(filename, source):
    if source == "SRBDF":
        match = re.search(r"(\d{8})", filename)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d")
    elif source == "ASCAT":
        match = re.search(r"Ant(\d{2})-(\d{3})", filename)
        if match:
            year = 2000 + int(match.group(1))
            doy = int(match.group(2))
            return datetime(year, 1, 1) + timedelta(days=doy - 1)
    elif source == "Sentinel":
        match = re.search(r"(\d{8})T", filename)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d")
    return None

# Process a folder and extract time series data
def process_folder(folder, source_label):
    results = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".tif"):
            continue
        date = parse_date_from_filename(filename, source_label)
        if date is None:
            continue
        path = os.path.join(folder, filename)
        value = extract_pixel_value(path, target_point[0], target_point[1])
        if value is not None:
            results.append((date, value))
    return sorted(results)

# Read CSV file and match values to Sentinel-1 acquisition dates
def read_csv_matched_to_sentinel(csv_path, sentinel_dates_set):
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date_str = row.get('date')
                if not date_str:
                    continue
                date = parser.parse(date_str).replace(hour=0, minute=0, second=0, microsecond=0)
                if date in sentinel_dates_set:
                    val_str = row.get('HH_dB')
                    if val_str is None or val_str == '':
                        continue
                    value = float(val_str)
                    data.append((date, value))
            except Exception:
                continue
    return sorted(data)

# Load data from SRBDF, ASCAT and Sentinel-1
srbdf_data = process_folder(srbdf_folder, "SRBDF")
ascat_data = process_folder(ascat_folder, "ASCAT")    
sentinel_data = process_folder(sentinel_folder, "Sentinel")

# Sentinel-1 dates set for CSV matching
sentinel_dates_set = set(d.replace(hour=0, minute=0, second=0, microsecond=0) for d, v in sentinel_data)

# Load external CSV data
csv_path = r"data:\Reference\Sentinel-1.csv"
csv_data = read_csv_matched_to_sentinel(csv_path, sentinel_dates_set)

# Convert list of tuples to dictionary {date: value}
def to_date_dict(data):
    return {d.date(): v for d, v in data}

srbdf_dict = to_date_dict(srbdf_data)
ascat_dict = to_date_dict(ascat_data)
sentinel_dict = to_date_dict(sentinel_data)
csv_dict = to_date_dict(csv_data)

# Find common acquisition dates
common_dates_all = set(srbdf_dict) & set(sentinel_dict) & set(csv_dict)

# Filter datasets by common dates
def filter_data_by_dates(data, date_set):
    return [(d, v) for d, v in data if d.date() in date_set]

srbdf_data_filtered = filter_data_by_dates(srbdf_data, common_dates_all)
sentinel_data_filtered = filter_data_by_dates(sentinel_data, common_dates_all)
csv_data_filtered = filter_data_by_dates(csv_data, common_dates_all)

# Prepare values for correlation calculation
srbdf_vals_all = [srbdf_dict[d] for d in sorted(common_dates_all)]
ascat_vals_all = [ascat_dict[d] for d in sorted(common_dates_all) if d in ascat_dict]
sentinel_vals_all = [sentinel_dict[d] for d in sorted(common_dates_all)]
csv_vals_all = [csv_dict[d] for d in sorted(common_dates_all)]

# Safe Pearson correlation (avoids crash if too few data points)
def safe_pearson(x, y):
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return pearsonr(x, y)[0]

# Calculate correlations
corr_srbdf_ascat = safe_pearson(srbdf_vals_all, ascat_vals_all)
corr_srbdf_sentinel = safe_pearson(srbdf_vals_all, sentinel_vals_all)
corr_ascat_sentinel = safe_pearson(ascat_vals_all, sentinel_vals_all)
corr_ascat_csv = safe_pearson(ascat_vals_all, csv_vals_all)

print("\nPearson correlation coefficients (intersection with CSV data):")
print(f"SRBDF vs ASCAT:      {corr_srbdf_ascat:.3f}")
print(f"SRBDF vs Sentinel-1: {corr_srbdf_sentinel:.3f}")
print(f"ASCAT vs Sentinel-1: {corr_ascat_sentinel:.3f}")
print(f"ASCAT vs CSV:        {corr_ascat_csv:.3f}")

# Plot configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(6, 3))
ax = plt.gca()

# Configure x-axis ticks
ax.xaxis.set_major_locator(mdates.DayLocator(interval=60))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Scatter plots for each dataset
if csv_data_filtered:
    dates, values = zip(*csv_data_filtered)
    plt.scatter(dates, values, label="Sentinel-1 (Raw)", color='grey', marker='o', s=12)

if srbdf_data_filtered:
    dates, values = zip(*srbdf_data_filtered)
    plt.scatter(dates, values, label="Sentinel-1 (Normalized, SRBDF)", color='red', marker='o', s=12)

if ascat_data:
    dates, values = zip(*ascat_data)
    plt.scatter(dates, values, label="ASCAT", color='royalblue', marker='^', s=12)

# Custom legends
legend_raw = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=8, label='Sentinel-1 (Raw)')
legend_norm = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Sentinel-1 (Normalized, SRBDF)')
legend_ascat = mlines.Line2D([], [], color='royalblue', marker='^', linestyle='None', markersize=8, label='ASCAT')

plt.xlabel("Date")
plt.ylabel("Backscatter Coefficient (dB)", fontsize=12)
plt.legend(handles=[legend_raw, legend_norm, legend_ascat], loc="lower left", fontsize=10, frameon=True)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.grid(True)

plt.tight_layout()
plt.savefig('time_series_comparison_final.png', dpi=600, bbox_inches='tight')
plt.show()
