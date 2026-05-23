import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from pyproj import Transformer
from sklearn.metrics import accuracy_score, cohen_kappa_score

aws_file_14 = "./AWS/IMAU_ANT_AWS14.tab"
base_dir = "./meltarea"

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "font.family": "Arial"
})

AWS_LAT = -67.0138
AWS_LON = -61.3967

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
x_aws, y_aws = transformer.transform(AWS_LON, AWS_LAT)


def read_aws(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if len(line.strip().split('\t')) > 10:
            header_line = i
            break

    df = pd.read_csv(file_path, sep='\t', skiprows=header_line,
                     engine='python', on_bad_lines='skip')
    df = df.dropna(axis=1, how='all')

    time_col = next(col for col in df.columns
                    if 'date' in str(col).lower() or 'time' in str(col).lower())

    temp_col = "TTT [°C] (corrected at 2m height)"

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

    df = df.dropna(subset=[time_col, temp_col])
    df = df[(df[temp_col] > -100) & (df[temp_col] < 20)]

    daily = df.groupby(df[time_col].dt.date)[temp_col].max().reset_index()
    daily.columns = ['date', 'temp']
    daily['date'] = pd.to_datetime(daily['date'])

    return daily


aws14 = read_aws(aws_file_14)

start_date = pd.to_datetime('2018-11-01')
end_date = pd.to_datetime('2019-03-31')

srbdf_folder = os.path.join(base_dir, "2019", "SRBDF")
s1_folder = os.path.join(base_dir, "2019", "Sentinel-1")
ascat_folder = os.path.join(base_dir, "2019", "ASCAT")

srbdf_files = sorted(glob.glob(os.path.join(srbdf_folder, "*.tif")))
s1_files = sorted(glob.glob(os.path.join(s1_folder, "*.tif")))
ascat_files = sorted(glob.glob(os.path.join(ascat_folder, "*.tif")))

df_srbdf, df_s1, df_ascat = [], [], []

def extract_melt_state(file):
    with rasterio.open(file) as src:
        data = src.read(1)
        transform = src.transform

        col, row = ~transform * (x_aws, y_aws)
        col = int(col)
        row = int(row)

        if row < 0 or col < 0 or row >= data.shape[0] or col >= data.shape[1]:
            return np.nan

        pixel = data[row, col]
        return 1 if pixel == 1 else 0


for f in srbdf_files:
    date = pd.to_datetime(os.path.basename(f).split('.')[0])
    if start_date <= date <= end_date:
        df_srbdf.append([date, extract_melt_state(f)])

for f in s1_files:
    try:
        date = pd.to_datetime(os.path.basename(f).split('.')[0])
        if start_date <= date <= end_date:
            df_s1.append([date, extract_melt_state(f)])
    except:
        continue

for f in ascat_files:
    date = pd.to_datetime(os.path.basename(f).split('.')[0])
    if start_date <= date <= end_date:
        df_ascat.append([date, extract_melt_state(f)])


df_srbdf = pd.DataFrame(df_srbdf, columns=['date', 'SRBDF'])
df_s1 = pd.DataFrame(df_s1, columns=['date', 'Sentinel-1'])
df_ascat = pd.DataFrame(df_ascat, columns=['date', 'ASCAT'])

df_all = df_srbdf.merge(df_ascat, on='date', how='outer')

if not df_s1.empty:
    df_all = df_all.merge(df_s1, on='date', how='outer')

df_all = df_all.sort_values('date')

aws14 = aws14[(aws14['date'] >= start_date) & (aws14['date'] <= end_date)]
aws_eval = aws14.copy()
aws_eval['AWS_melt'] = (aws_eval['temp'] >= -0.8).astype(int)

df_eval = aws_eval.merge(df_all, on='date', how='inner')

metrics_summary = []

for col in ['SRBDF', 'ASCAT', 'Sentinel-1']:
    if col in df_eval.columns:
        valid = df_eval[['AWS_melt', col]].dropna()

        if len(valid) > 0 and len(valid['AWS_melt'].unique()) > 1:
            oa = accuracy_score(valid['AWS_melt'], valid[col])
            kappa = cohen_kappa_score(valid['AWS_melt'], valid[col])

            metrics_summary.append(
                f"{col} vs AWS:\n  OA = {oa:.3f}\n  Kappa = {kappa:.3f}"
            )
        else:
            metrics_summary.append(f"{col} vs AWS:\n  Insufficient data")

print("\n".join(metrics_summary))

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(aws14['date'], aws14['temp'],
        color='steelblue', linewidth=2,
        label='AWS14 Temperature')

ax.axhline(-0.8, color='black', linestyle='--',
           linewidth=1.5, label='Melt Threshold (-0.8°C)')

ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3)

y_lim_bottom, y_lim_top = ax.get_ylim()
ax.axhspan(-0.8, y_lim_top,
           color='#F7E6BA', alpha=0.4)

ax.set_ylim(y_lim_bottom, y_lim_top)

ax2 = ax.twinx()

ax2.plot(df_all['date'], df_all['SRBDF'],
         color='#FF3B30', linewidth=2.2, label='SRBDF')

ax2.plot(df_all['date'], df_all['ASCAT'],
         color="#9970AC", linestyle='--', linewidth=2,
         label='ASCAT')

if not df_s1.empty:
    ax2.scatter(df_s1['date'], df_s1['Sentinel-1'],
                color='#629641', s=22,
                label='Sentinel-1')

ax2.set_ylim(-0.1, 1.1)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([])
ax2.set_yticklabels([])

ax.text(0.98, 0.92, "Melt",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=26, fontweight='bold',
        color='#8F1B26')

ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(lines1 + lines2,
          labels1 + labels2,
          loc='center',
          bbox_to_anchor=(0.66, 0.24),
          frameon=True,
          edgecolor='black',
          prop={'size': 19})

plt.tight_layout()
plt.show()
