import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from pyproj import Transformer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from matplotlib.patches import Patch

current_dir = os.getcwd()

aws_file_14 = os.path.join(current_dir, "IMAU_ANT_AWS14.tab")
base_dir = current_dir

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

    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=header_line,
        engine='python',
        on_bad_lines='skip'
    )

    df = df.dropna(axis=1, how='all')

    time_col = next(
        col for col in df.columns
        if 'date' in str(col).lower() or 'time' in str(col).lower()
    )

    temp_col = "TTT [°C]"

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

    df = df.dropna(subset=[time_col, temp_col])
    df = df[(df[temp_col] > -100) & (df[temp_col] < 20)]

    daily = df.groupby(df[time_col].dt.date)[temp_col].max().reset_index()

    daily.columns = ['date', 'temp']
    daily['date'] = pd.to_datetime(daily['date'])

    return daily

aws14 = read_aws(aws_file_14)

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

start_date = pd.to_datetime('2018-11-01')
end_date = pd.to_datetime('2019-03-31')

srbdf_folder = os.path.join(base_dir, '2019', "SRBDF")
s1_folder = os.path.join(base_dir, '2019', "Sentinel-1")
ascat_folder = os.path.join(base_dir, '2019', "ASCAT")

srbdf_files = sorted(glob.glob(os.path.join(srbdf_folder, "*.tif")))
s1_files = sorted(glob.glob(os.path.join(s1_folder, "*.tif")))
ascat_files = sorted(glob.glob(os.path.join(ascat_folder, "*.tif")))

df_srbdf = []
df_s1 = []
df_ascat = []

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

full_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq='D'
)

df_all = pd.DataFrame({'date': full_dates})

df_all = df_all.merge(df_srbdf, on='date', how='left')
df_all = df_all.merge(df_ascat, on='date', how='left')

if not df_s1.empty:
    df_all = df_all.merge(df_s1, on='date', how='left')
else:
    df_all['Sentinel-1'] = np.nan

df_all = df_all.sort_values('date')

aws14 = aws14[
    (aws14['date'] >= start_date) &
    (aws14['date'] <= end_date)
]

aws_eval = aws14.copy()

aws_eval['AWS_melt'] = (
    aws_eval['temp'] >= -0.8
).astype(int)

df_eval = aws_eval.merge(df_all, on='date', how='inner')

metrics_summary = []

for col in ['SRBDF', 'ASCAT', 'Sentinel-1']:

    if col in df_eval.columns:

        valid = df_eval[['AWS_melt', col]].dropna()

        if len(valid) > 0 and len(valid['AWS_melt'].unique()) > 1:

            oa = accuracy_score(valid['AWS_melt'], valid[col])

            kappa = cohen_kappa_score(
                valid['AWS_melt'],
                valid[col]
            )

            metrics_summary.append(
                f"{col} vs AWS:\n"
                f"  OA = {oa:.3f}\n"
                f"  Kappa = {kappa:.3f}"
            )

        else:
            metrics_summary.append(
                f"{col} vs AWS:\n"
                f"  Insufficient data"
            )

print("\n".join(metrics_summary))

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(
    aws14['date'],
    aws14['temp'],
    color='steelblue',
    linewidth=2,
    label='AWS14 Temperature'
)

ax.axhline(
    -0.8,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label='Melt Threshold (-0.8°C)'
)

ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3)

y_lim_bottom, y_lim_top = ax.get_ylim()

bar_start_y = y_lim_top + 0.3
bar_height = 1.2
bar_gap = 0.6

ax.set_ylim(
    y_lim_bottom,
    bar_start_y + (bar_height + bar_gap) * 3 + 0.5
)

bar_configs = [
    ('SRBDF', '#E41A1C', bar_start_y),
    ('ASCAT', '#FF7F00', bar_start_y + bar_height + bar_gap),
    ('Sentinel-1', '#178642', bar_start_y + (bar_height + bar_gap) * 2)
]

plot_dates = mdates.date2num(df_all['date'])

for col_name, melt_color, base_y in bar_configs:

    state = df_all[col_name].values

    for i in range(len(plot_dates)):

        if np.isnan(state[i]):
            continue

        color = melt_color if state[i] == 1 else '#D3D3D3'

        ax.fill_between(
            [plot_dates[i] - 0.5, plot_dates[i] + 0.5],
            base_y,
            base_y + bar_height,
            color=color,
            linewidth=0
        )

ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d')
)

plt.setp(
    ax.get_xticklabels(),
    rotation=0,
    ha='center'
)

legend_elements = [
    plt.Line2D(
        [0],
        [0],
        color='steelblue',
        lw=2,
        label='AWS14 Temperature'
    ),

    plt.Line2D(
        [0],
        [0],
        color='black',
        linestyle='--',
        lw=1.5,
        label='Melt Threshold (-0.8°C)'
    ),

    Patch(
        facecolor='#E41A1C',
        edgecolor='none',
        label='SRBDF'
    ),

    Patch(
        facecolor='#FF7F00',
        edgecolor='none',
        label='ASCAT'
    ),

    Patch(
        facecolor='#178642',
        edgecolor='none',
        label='Sentinel-1'
    ),

    Patch(
        facecolor='#D3D3D3',
        edgecolor='none',
        label='Freeze'
    )
]

ax.legend(
    handles=legend_elements,
    loc='lower right',
    bbox_to_anchor=(0.60, 0.05),
    frameon=True,
    edgecolor='black',
    ncol=1,
    prop={'size': 18}
)

plt.tight_layout()
plt.show()
