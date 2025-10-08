import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from datetime import datetime

# Base directory where data is stored
base_dir = r"data:\meltarea"

# Years to be analyzed
years = ['2019', '2020', '2021']

# Datasets to be compared: ASCAT, SRBDF, Sentinel-1
datasets = ['ASCAT', 'SRBDF', 'Sentinel-1']

# Pixel area in km² (assume 0.2 km × 0.2 km pixels = 0.04 km²)
pixel_area = 0.04  

# DataFrame to collect all results
df_all = pd.DataFrame()

def compute_melt_area(file, pixel_size):
    """
    Compute melt area from binary raster.
    Pixels with value == 1 are considered as melt pixels.
    """
    with rasterio.open(file) as src:
        data = src.read(1)
        melt_area_km2 = np.sum(data == 1) * pixel_size
    return melt_area_km2

# Loop through each year and dataset
for year in years:
    dfs_year = []
    for dataset in datasets:
        folder = os.path.join(base_dir, year, dataset)
        files = sorted(glob.glob(os.path.join(folder, "*.tif")))
        if not files:
            continue
        dates = [os.path.basename(f).split('.')[0] for f in files]
        values = [compute_melt_area(f, pixel_area) for f in files]

        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            dataset: values
        })
        dfs_year.append(df)

    # Merge datasets for each year
    if len(dfs_year) == 3:
        merged = dfs_year[0]
        for df in dfs_year[1:]:
            merged = pd.merge(merged, df, on='date', how='outer')
        merged['year'] = year
        df_all = pd.concat([df_all, merged], ignore_index=True)

# Select winter months only (Nov–Mar)
df_all = df_all[df_all['date'].dt.month.isin([11, 12, 1, 2, 3])]

def assign_season(row):
    """
    Assign winter season label, e.g., 2019-2020
    based on the date.
    """
    month = row['date'].month
    year = row['date'].year
    if month >= 11:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"

df_all['season'] = df_all.apply(assign_season, axis=1)

# Matplotlib settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 16

# Define style for each dataset
style_dict = {
    'SRBDF': {'color': "#1D8D7C", 'marker': 'o', 'linestyle': '-', 'label': 'SRBDF'},
    'ASCAT': {'color': '#D57B70', 'marker': 's', 'linestyle': '--', 'label': 'ASCAT'},
    'Sentinel-1': {'color': '#1C3C63', 'marker': '^', 'linestyle': '-.', 'label': 'Sentinel-1'}
}

# Plot melt area time series for each year
fig, axes = plt.subplots(len(years), 1, figsize=(9, 14), sharex=False, sharey=True)

for i, year in enumerate(years):
    ax = axes[i]
    data_year = df_all[df_all['year'] == year]
    if data_year.empty:
        continue

    # Set x-axis range
    x_min = data_year['date'].min() - pd.Timedelta(days=2)
    x_max = data_year['date'].max() + pd.Timedelta(days=2)
    ax.set_xlim(x_min, x_max)

    if i == len(years)//2: 
       ax.set_ylabel('Melt Area (km²)', fontsize=24)

    # Plot each dataset
    for dataset in datasets:
        label = style_dict[dataset]['label'] if i == 0 else ""
        markersize = 10 if dataset == 'Sentinel-1' else 4  

        ax.plot(data_year['date'], data_year[dataset],
                label=label,
                color=style_dict[dataset]['color'],
                marker=style_dict[dataset]['marker'],
                linestyle=style_dict[dataset]['linestyle'],
                markersize=markersize,
                linewidth=1.5,
                alpha=0.9)

    ax.grid(False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=0, labelsize=22, pad=1.5)
    ax.tick_params(axis='y', labelsize=22) 

    # Customize axis border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Add legend (only once at the top panel)
    if i == 0:
        legend_elements = [
            Line2D([0], [0], color=style_dict['SRBDF']['color'], marker=style_dict['SRBDF']['marker'],
                   linestyle=style_dict['SRBDF']['linestyle'], label=style_dict['SRBDF']['label'], markersize=6),
            Line2D([0], [0], color=style_dict['ASCAT']['color'], marker=style_dict['ASCAT']['marker'],
                   linestyle=style_dict['ASCAT']['linestyle'], label=style_dict['ASCAT']['label'], markersize=6),
            Line2D([0], [0], color=style_dict['Sentinel-1']['color'], marker=style_dict['Sentinel-1']['marker'],
                   linestyle='None', label=style_dict['Sentinel-1']['label'], markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, 1.0),
                  fontsize=20, frameon=True, fancybox=True,
                  edgecolor='black', markerscale=2.0)

axes[-1].set_xlabel('Date', fontsize=24)

fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.96, hspace=0.20)
fig.savefig('melt_area_vertical_boxed.png', dpi=1200, bbox_inches='tight', pad_inches=0.2)
plt.show()

def compute_rmsd(y_true, y_pred):
    """Compute root mean square deviation (RMSD)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# List of all winter seasons
seasons = sorted(df_all['season'].unique())

print("Correlation, RMSD, MAE and percentage errors for each winter season:")
for season in seasons:
    data_season = df_all[df_all['season'] == season]
    print(f"\nWinter season {season} correlation matrix:")
    print(data_season[datasets].corr().round(3))

    avg_area = data_season[datasets].mean().mean()
    print(f"Mean melt area during {season}: {avg_area:.2f} km²")

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1 = datasets[i]
            ds2 = datasets[j]
            valid = data_season[[ds1, ds2]].dropna()
            if valid.empty:
                print(f"{ds1} vs {ds2}: no valid data")
                continue
            rmsd = compute_rmsd(valid[ds1], valid[ds2])
            mae = mean_absolute_error(valid[ds1], valid[ds2])
            rmsd_pct = rmsd / avg_area * 100
            mae_pct = mae / avg_area * 100
            print(f"{ds1} vs {ds2}: RMSD = {rmsd:.2f} km² ({rmsd_pct:.2f}%), "
                  f"MAE = {mae:.2f} km² ({mae_pct:.2f}%)")

print("\nOverall correlation, RMSD, MAE and percentage errors for all seasons:")
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        ds1 = datasets[i]
        ds2 = datasets[j]

        valid_list = []
        for season in seasons:
            data_season = df_all[df_all['season'] == season]
            valid = data_season[[ds1, ds2]].dropna()
            valid_list.append(valid)

        all_valid_data = pd.concat(valid_list, ignore_index=True)

        avg_area_all = all_valid_data[[ds1, ds2]].mean().mean()
        rmsd_all = compute_rmsd(all_valid_data[ds1], all_valid_data[ds2])
        mae_all = mean_absolute_error(all_valid_data[ds1], all_valid_data[ds2])
        r_value, _ = pearsonr(all_valid_data[ds1], all_valid_data[ds2])

        rmsd_pct_all = rmsd_all / avg_area_all * 100
        mae_pct_all = mae_all / avg_area_all * 100

        print(f"{ds1} vs {ds2}: r = {r_value:.3f}, RMSD = {rmsd_all:.2f} km² ({rmsd_pct_all:.2f}%), "
              f"MAE = {mae_all:.2f} km² ({mae_pct_all:.2f}%)")
