import numpy as np
import rasterio
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

base_dir = os.getcwd()

aws_file_18 = os.path.join(base_dir, "IMAU_ANT_AWS18.tab")
aws_file_14 = os.path.join(base_dir, "IMAU_ANT_AWS14.tab")

years = ['2019', '2020', '2021']
pixel_area = 0.04

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "font.family": "Arial"
})

def compute_melt_area(file, pixel_size):
    with rasterio.open(file) as src:
        data = src.read(1)
        return np.sum(data == 1) * pixel_size

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

    temp_col = "TTT [°C] (corrected at 2m height)"

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

    df = df.dropna(subset=[time_col, temp_col])

    df = df[
        (df[temp_col] > -100) &
        (df[temp_col] < 20)
    ]

    daily = df.groupby(df[time_col].dt.date)[temp_col].max().reset_index()

    daily.columns = ['date', 'temp']

    daily['date'] = pd.to_datetime(daily['date'])

    return daily

aws18 = read_aws(aws_file_18)
aws14 = read_aws(aws_file_14)

df_all = pd.DataFrame()

for year in years:

    folder = os.path.join(base_dir, year, "SRBDF")

    files = sorted(glob.glob(os.path.join(folder, "*.tif")))

    if not files:
        continue

    dates = [os.path.basename(f).split('.')[0] for f in files]

    values = [compute_melt_area(f, pixel_area) for f in files]

    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'SRBDF': values
    })

    df['year'] = year

    df_all = pd.concat([df_all, df], ignore_index=True)

df_all = df_all[df_all['date'].dt.month.isin([11, 12, 1, 2, 3])]

def assign_season(row):

    month = row['date'].month
    year = row['date'].year

    if month >= 11:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"

df_all['season'] = df_all.apply(assign_season, axis=1)

seasons = [
    ('2018-11-01', '2019-03-31'),
    ('2019-11-01', '2020-03-31'),
    ('2020-11-01', '2021-03-31')
]

def plot_figure(threshold, output_name):

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for i, (ax, (start, end)) in enumerate(zip(axes, seasons)):

        d18 = aws18[
            (aws18['date'] >= start) &
            (aws18['date'] <= end)
        ]

        d14 = aws14[
            (aws14['date'] >= start) &
            (aws14['date'] <= end)
        ]

        melt = df_all[
            (df_all['date'] >= start) &
            (df_all['date'] <= end)
        ]

        ax.plot(
            d18['date'],
            d18['temp'],
            color='blue',
            linewidth=1.8,
            label='AWS18 Daily Max Temperature'
        )

        ax.plot(
            d14['date'],
            d14['temp'],
            color='green',
            linewidth=1.8,
            label='AWS14 Daily Max Temperature'
        )

        ax.axhline(
            y=threshold,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Threshold ({threshold}°C)'
        )

        ax.set_ylabel("Temperature (°C)")

        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()

        ax2.plot(
            melt['date'],
            melt['SRBDF'],
            color='#E24A33',
            linewidth=1.8,
            label='SRBDF Melt Area'
        )

        ax2.set_ylabel("Melt Area (km²)")

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m-%d')
        )

        plt.setp(
            ax.get_xticklabels(),
            rotation=0,
            ha='center'
        )

        if i == 2:

            lines1, labels1 = ax.get_legend_handles_labels()

            lines2, labels2 = ax2.get_legend_handles_labels()

            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.25),
                frameon=True,
                edgecolor='black'
            )

    axes[-1].set_xlabel("Date")

    plt.tight_layout()

    plt.savefig(
        output_name,
        dpi=900,
        bbox_inches='tight'
    )

    plt.show()

plot_figure(-0.8, "SRBDF_AWS_Threshold_-0.8.png")
plot_figure(0, "SRBDF_AWS_Threshold_0.png")
