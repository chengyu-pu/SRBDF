import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Step 1: Read input data
# ===========================
# Input Excel file must contain at least four columns:
# "date" (YYYY-MM format), "ASCAT", "S1A", "S1B"
file_path = r'D:\date.xlsx'
df = pd.read_excel(file_path)

# Convert "date" column to string (year-month format)
months = df['date'].astype(str)

# ===========================
# Step 2: Format x-axis labels
# ===========================
formatted_labels = []   # labels displayed on x-axis
is_year_label = []      # True if label should include year
prev_year = ''
for m in months:
    year, month = m.split('-')
    if year != prev_year:   # first month of each year shows full "YYYY-MM"
        formatted_labels.append(f'{year}-{month}')
        is_year_label.append(True)
        prev_year = year
    else:                   # subsequent months show only "MM"
        formatted_labels.append(month)
        is_year_label.append(False)

# ===========================
# Step 3: Extract data series
# ===========================
ascat_values = df['ASCAT'].values
s1a_values = df['S1A'].values
s1b_values = df['S1B'].values

# ===========================
# Step 4: Plot bar chart
# ===========================
fig, ax = plt.subplots(figsize=(8.0, 3.94))

x = np.arange(len(months))  # x-axis index
bar_width = 0.35            # width of each bar
offset = bar_width / 4      # shift for alignment

# Define consistent colors for different sensors
colors = {
    'ASCAT': '#EED26E',     # Yellow
    'S1A': '#2356A7',       # Blue
    'S1B': '#9CC7F1'        # Light blue
}

# Draw ASCAT bars (left-aligned)
ax.bar(x - bar_width / 2 - offset, ascat_values, width=bar_width,
       label='ASCAT', color=colors['ASCAT'])

# Draw Sentinel-1A and Sentinel-1B stacked bars (right-aligned)
s1_base = s1a_values
ax.bar(x + bar_width / 2 - offset, s1a_values, width=bar_width,
       label='S1A', color=colors['S1A'])
ax.bar(x + bar_width / 2 - offset, s1b_values, width=bar_width,
       label='S1B', color=colors['S1B'], bottom=s1_base)

# Remove default legend (will be exported separately)
ax.legend().remove()

# ===========================
# Step 5: Customize x-axis labels
# ===========================
ax.set_xticks(x)
ax.set_xticklabels([''] * len(x))  # hide default ticks

# Place custom rotated text labels under the axis
for i, label in enumerate(formatted_labels):
    if is_year_label[i]:
        ax.text(x[i] - 1.0, -0.8, label, rotation=45,
                ha='left', va='top', fontsize=18)
    else:
        ax.text(x[i], -0.8, label, rotation=45,
                ha='center', va='top', fontsize=18)

ax.set_xlim(-1.2, len(x) - 0.3)

# ===========================
# Step 6: Axis style
# ===========================
ax.set_ylabel('Number of Images', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

# ===========================
# Step 7: Export separate legend
# ===========================
fig_legend = plt.figure(figsize=(8, 1.0))
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")

# Create legend handles (rectangles with corresponding colors)
handles = [
    plt.Rectangle((0,0),1,1, color=colors['ASCAT'], label="ASCAT"),
    plt.Rectangle((0,0),1,1, color=colors['S1A'], label="Sentinel-1A"),
    plt.Rectangle((0,0),1,1, color=colors['S1B'], label="Sentinel-1B")
]

# Place legend horizontally
legend_ax.legend(
    handles=handles,
    loc='center',
    fontsize=20,
    frameon=False,   # remove frame
    ncol=3           # arrange in three columns
)

# ===========================
# Step 8: Save figures
# ===========================
fig.savefig('ASCAT_vs_Sentinel1_stacked_S1A_S1B.png',
            dpi=1200, bbox_inches='tight')
fig_legend.savefig('Legend_ASC_S1A_S1B.png',
                   dpi=1200, bbox_inches='tight')

plt.show()
