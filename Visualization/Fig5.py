import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from rasterio.warp import reproject, Resampling
from matplotlib.ticker import ScalarFormatter

# ======================================================
# Step 1: Define paths and global parameters
# ======================================================
folder = "data/standard_deviation"   # Folder containing input .tif files
reference_path = "data/reference/reference.tif"

# Plot colors
color_original = "#C2BAAC"
color_normalized = '#1f77b4'

# ======================================================
# Step 2: Define helper functions
# ======================================================
def read_tiff(path):
    """Read a single-band GeoTIFF file and return array + metadata profile."""
    with rasterio.open(path) as src:
        return src.read(1), src.profile

def align_to_reference(data, profile, ref_path):
    """Reproject input data to match the grid of a reference GeoTIFF."""
    with rasterio.open(ref_path) as ref:
        ref_data = np.empty((ref.height, ref.width), dtype=data.dtype)
        reproject(
            source=data,
            destination=ref_data,
            src_transform=profile['transform'],
            src_crs=profile['crs'],
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest
        )
    return ref_data

def get_valid_mask(reference_path):
    """Create a validity mask (exclude NoData) from reference raster."""
    with rasterio.open(reference_path) as ref:
        mask = ref.read(1)
        return mask != 0

# ======================================================
# Step 3: Prepare reference profile and mask
# ======================================================
valid_mask = get_valid_mask(reference_path)
with rasterio.open(reference_path) as ref:
    ref_profile = ref.profile

# ======================================================
# Step 4: Parse input filenames
# ======================================================
# Filenames must match: (ASC|DESC)_YYYYMMDD_(originalHH|normalizedHH).tif
file_list = os.listdir(folder)
pattern = r'(ASC|DESC)_(\d{8})_(originalHH|normalizedHH)\.tif'
matched = {}

for f in file_list:
    match = re.match(pattern, f)
    if match:
        orbit, date, dtype = match.groups()
        key = f'{orbit}_{date}'
        if key not in matched:
            matched[key] = {}
        matched[key][dtype] = os.path.join(folder, f)

# ======================================================
# Step 5: Process and align paired images
# ======================================================
all_original = []
all_normalized = []

for key, paths in matched.items():
    if 'originalHH' in paths and 'normalizedHH' in paths:
        print(f'Processing: {key}')
        img_o, prof_o = read_tiff(paths['originalHH'])
        img_n, prof_n = read_tiff(paths['normalizedHH'])

        # Ensure CRS and transform metadata are available
        if prof_o.get('crs') is None or prof_o.get('transform') is None:
            print(f"  → originalHH missing CRS/transform — using reference")
            prof_o['crs'] = ref_profile['crs']
            prof_o['transform'] = ref_profile['transform']

        if prof_n.get('crs') is None or prof_n.get('transform') is None:
            print(f"  → normalizedHH missing CRS/transform — using reference")
            prof_n['crs'] = ref_profile['crs']
            prof_n['transform'] = ref_profile['transform']

        # Reproject to reference grid
        img_o = align_to_reference(img_o, prof_o, reference_path)
        img_n = align_to_reference(img_n, prof_n, reference_path)

        # Apply valid mask and remove NaNs
        mask = valid_mask & ~np.isnan(img_o) & ~np.isnan(img_n)

        all_original.append(img_o[mask])
        all_normalized.append(img_n[mask])

# Concatenate all pixels across scenes
all_original = np.concatenate(all_original)
all_normalized = np.concatenate(all_normalized)

# ======================================================
# Step 6: Plot histograms of backscatter distributions
# ======================================================
plt.figure(figsize=(7.2, 3.54))
plt.hist(all_original, bins=200, alpha=1, label='Original', color=color_original)
plt.hist(all_normalized, bins=200, alpha=1, label='Normalized', color=color_normalized)
plt.xlim(-30, 10)
plt.ylim(0, 12000000)
plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Backscatter Coefficient (dB)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Format y-axis in scientific notation
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('histogram_all.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================
# Step 7: Compare standard deviation before/after normalization
# ======================================================
std_o = np.nanstd(all_original)
std_n = np.nanstd(all_normalized)

plt.figure(figsize=(2.5, 4))
bar_width = 0.2
x_positions = [0, 0.4]  

bars = plt.bar(x_positions, [std_o, std_n], width=bar_width, 
               color=[color_original, color_normalized], alpha=1)

# Annotate bars with numeric values
for bar, val in zip(bars, [std_o, std_n]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha='center', va='bottom', fontsize=12)

plt.xticks(x_positions, ['Original', 'Normalized'], fontsize=12)
plt.ylabel('Standard Deviation (dB)', fontsize=12)
plt.ylim(7.0, 7.8)
plt.yticks(np.arange(7.0, 7.81, 0.2), fontsize=12)
plt.tight_layout()
plt.savefig('std_bar_chart_overall.png', dpi=1200)
plt.show()
