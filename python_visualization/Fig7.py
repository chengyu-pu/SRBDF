import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

# -----------------------------------------------------------------------------
# Configuration: define data directories
# -----------------------------------------------------------------------------
srbdf_folder = "data/SRBDF"
ascat_folder = "data/ASCAT"
sentinel_folder = "data/Sentinel-1"

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def extract_all_pixel_values(tif_path):
    """
    Extract all valid pixel values from a GeoTIFF file.
    
    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF file.
    
    Returns
    -------
    numpy.ndarray
        1D array of valid pixel values (NaN and nodata excluded).
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        mask = (data != src.nodata) if src.nodata is not None else ~np.isnan(data)
        return data[mask]


def parse_date_from_filename(filename, source):
    """
    Parse acquisition date from the file name for different data sources.
    
    Parameters
    ----------
    filename : str
        File name to parse.
    source : str
        Data source ("SRBDF", "ASCAT", or "Sentinel").
    
    Returns
    -------
    datetime.datetime or None
        Parsed date object or None if parsing fails.
    """
    if source == "SRBDF":
        match = re.search(r"(\d{8})", filename)
        if match:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d")
    elif source == "ASCAT":
        match = re.search(r"Ant(\d{2})-(\d{3})", filename)
        if match:
            year = 2000 + int(match.group(1))
            doy = int(match.group(2))
            return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
    elif source == "Sentinel":
        match = re.search(r"(\d{8})T", filename)
        if match:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d")
    return None


def process_folder(folder, source_label):
    """
    Process a folder of GeoTIFFs for one data source, extracting pixel values
    and associated acquisition dates.
    
    Parameters
    ----------
    folder : str
        Folder containing GeoTIFF files.
    source_label : str
        Data source label ("SRBDF", "ASCAT", "Sentinel").
    
    Returns
    -------
    list of (datetime, numpy.ndarray)
        Sorted list of (date, pixel values).
    """
    values = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".tif"):
            continue
        date = parse_date_from_filename(filename, source_label)
        if date is None:
            continue
        path = os.path.join(folder, filename)
        data_values = extract_all_pixel_values(path)
        values.append((date, data_values))
    values.sort(key=lambda x: x[0])
    return values


def to_dict(data):
    """Convert list of (date, values) into a dictionary."""
    return {d: v for d, v in data}


# -----------------------------------------------------------------------------
# Load data from the three sources
# -----------------------------------------------------------------------------
srbdf_data = process_folder(srbdf_folder, "SRBDF")
ascat_data = process_folder(ascat_folder, "ASCAT")
sentinel_data = process_folder(sentinel_folder, "Sentinel")

srbdf_dict = to_dict(srbdf_data)
ascat_dict = to_dict(ascat_data)
sentinel_dict = to_dict(sentinel_data)

# Identify common acquisition dates across all datasets
common_dates = set(srbdf_dict) & set(ascat_dict) & set(sentinel_dict)
srbdf_vals = [srbdf_dict[d] for d in sorted(common_dates)]
ascat_vals = [ascat_dict[d] for d in sorted(common_dates)]
sentinel_vals = [sentinel_dict[d] for d in sorted(common_dates)]

# Flatten into single arrays
srbdf_vals_flat = np.concatenate(srbdf_vals)
ascat_vals_flat = np.concatenate(ascat_vals)
sentinel_vals_flat = np.concatenate(sentinel_vals)

# Ensure equal lengths for comparison
min_len = min(len(srbdf_vals_flat), len(ascat_vals_flat), len(sentinel_vals_flat))
srbdf_vals_flat = srbdf_vals_flat[:min_len]
ascat_vals_flat = ascat_vals_flat[:min_len]
sentinel_vals_flat = sentinel_vals_flat[:min_len]


def clean_data(x, y):
    """Remove NaN/Inf pairs from two arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# -----------------------------------------------------------------------------
# Statistical evaluation (Pearson correlation & RMSE)
# -----------------------------------------------------------------------------
srbdf_ascat_x, srbdf_ascat_y = clean_data(srbdf_vals_flat, ascat_vals_flat)
srbdf_sentinel_x, srbdf_sentinel_y = clean_data(srbdf_vals_flat, sentinel_vals_flat)
ascat_sentinel_x, ascat_sentinel_y = clean_data(ascat_vals_flat, sentinel_vals_flat)


def safe_pearson(x, y):
    """Compute Pearson correlation safely (return NaN if insufficient samples)."""
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    return pearsonr(x, y)


corr_sa, _ = safe_pearson(srbdf_ascat_x, srbdf_ascat_y)
corr_ss, _ = safe_pearson(srbdf_sentinel_x, srbdf_sentinel_y)
corr_as, _ = safe_pearson(ascat_sentinel_x, ascat_sentinel_y)


def calculate_rmse(x, y):
    """Compute root-mean-square error (RMSE)."""
    return np.sqrt(np.mean((x - y) ** 2))


rmse_sa = calculate_rmse(srbdf_ascat_x, srbdf_ascat_y)
rmse_ss = calculate_rmse(srbdf_sentinel_x, srbdf_sentinel_y)
rmse_as = calculate_rmse(ascat_sentinel_x, ascat_sentinel_y)

# -----------------------------------------------------------------------------
# Visualization setup
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

vmin = -40
vmax = 15


def add_plot(x, y, xlabel, ylabel, r):
    """
    Generate a hexbin scatter plot comparing two datasets, 
    with linear regression and correlation statistics.
    """
    x, y = clean_data(x, y)
    coeffs = np.polyfit(x, y, 1)

    # Construct a custom colormap with white background for low frequencies
    freq_min = 1
    freq_white_limit = 10  
    freq_max = 10000  

    log_min = np.log10(freq_min)
    log_white = np.log10(freq_white_limit)
    log_max = np.log10(freq_max)

    white_frac = (log_white - log_min) / (log_max - log_min)
    n_colors = 256
    n_white = int(n_colors * white_frac)

    viridis = plt.get_cmap('viridis')
    viridis_colors = viridis(np.linspace(0, 1, n_colors - n_white))
    white_colors = np.ones((n_white, 4))
    colors = np.vstack((white_colors, viridis_colors))
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', colors)

    fig, ax = plt.subplots(figsize=(11 / 2.54, 11 / 2.54), dpi=300)

    # Hexbin density plot (log scale for frequency)
    hb = ax.hexbin(
        x, y, gridsize=60, cmap=white_viridis,
        mincnt=1, norm=LogNorm(vmin=freq_min, vmax=freq_max),
        alpha=0.95
    )

    # Add colorbar for frequency
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(hb, cax=cax)
    cb.set_label('Frequency (log scale)', fontsize=12)

    # Reference 1:1 line (black dashed)
    x_vals = np.linspace(vmin, vmax, 100)
    ax.plot(x_vals, x_vals, 'k--', label='_nolegend_')

    # Linear regression line (dark red dashed)
    y_fit = np.polyval(coeffs, x_vals)
    ax.plot(x_vals, y_fit, color='#8B1A1A', linestyle='--', label='_nolegend_')

    # Axis formatting
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(f"{xlabel} Backscatter Coefficient (dB)", fontsize=11)
    ax.set_ylabel(f"{ylabel} Backscatter Coefficient (dB)", fontsize=11)

    # Display regression equation and correlation in the lower right corner
    intercept_str = f"+ {coeffs[1]:.2f}" if coeffs[1] >= 0 else f"- {abs(coeffs[1]):.2f}"
    eq_text = f"y = {coeffs[0]:.2f}x {intercept_str}\nr = {r:.2f}"
    ax.text(0.95, 0.02, eq_text, transform=ax.transAxes,
            fontsize=11, color='#8B1A1A', verticalalignment='bottom', horizontalalignment='right')

    ax.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.show()


# -----------------------------------------------------------------------------
# Generate comparison plots
# -----------------------------------------------------------------------------
add_plot(srbdf_ascat_x, srbdf_ascat_y, "SRBDF", "ASCAT", corr_sa)
add_plot(srbdf_sentinel_x, srbdf_sentinel_y, "SRBDF", "Sentinel-1", corr_ss)
add_plot(ascat_sentinel_x, ascat_sentinel_y, "ASCAT", "Sentinel-1", corr_as)
