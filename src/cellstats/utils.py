import math
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import skimage.measure
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import label, regionprops_table
from sklearn.cluster import KMeans


def plot_image(image_path: Path, band: int = 0, band_color: str = 'red'):
    """Plot multi-band image with full RGB and single band visualization.

    Creates a side-by-side visualization showing the full RGB image and
    a single band with custom colormap.

    Args:
        image_path (Path): Path to the image file.
        band (int, optional): Band index to visualize separately. Defaults to 0.
        band_color (str, optional): Color for single band visualization.
            Defaults to 'red'.

    """
    cmap = LinearSegmentedColormap.from_list(
        'cmap', [(0, 'white'), (1, band_color)]
    )
    image_array = skimage.io.imread(image_path)
    _, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].matshow(image_array)
    axs[1].matshow(image_array[:, :, band], cmap=cmap, vmin=0, vmax=255)
    plt.show()


def plot_regions(
    image_path: Path,
    n_buffer: int = 2,
    n_limit: int = 25,
    band: int = 0,
    connectivity: int = 1,
):
    """Plot individual detected regions from an image.

    Detects regions in a specified band and displays them in a grid layout.
    Each region is shown with optional buffer pixels around it.

    Args:
        image_path (Path): Path to the image file.
        n_buffer (int, optional): Number of pixels to include around each region
            as buffer. Defaults to 2.
        n_limit (int, optional): Maximum number of regions to plot. Defaults to 25.
        band (int, optional): Band index to process for region detection.
            Defaults to 0.
        connectivity (int, optional): Connectivity for labeling (1 or 2).
            1 for 4-connectivity, 2 for 8-connectivity. Defaults to 1.

    """
    image_array = skimage.io.imread(image_path)
    band_array = image_array[:, :, band]
    band_label = skimage.measure.label(band_array, connectivity=connectivity)
    region_props = skimage.measure.regionprops(band_label)

    num = min(len(region_props), n_limit)
    nrows = math.ceil(math.sqrt(num))
    ncols = math.ceil(num / nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
    axs = axs.ravel()
    for i in range(num):
        if n_buffer > 0:
            slice_ = tuple(
                slice(
                    max(0, region_props[i].slice[dim].start - n_buffer),
                    region_props[i].slice[dim].stop + n_buffer,
                )
                for dim in range(2)
            )
        else:
            slice_ = region_props[i].slice
        array = image_array[slice_]
        axs[i].matshow(array)
        axs[i].set_title(
            f'Label: {region_props[i].label}'
            f'\nArea: {int(region_props[i].area)}'
        )
    fig.tight_layout()
    plt.show()


Properties: TypeAlias = Literal[
    'label', 'area', 'bbox', 'centroid', 'solidity', 'eccentricity'
]


def _create_regionprops_dataframe(
    image_path: Path,
    band: int = 0,
    connectivity: int = 1,
    properties: tuple[Properties, ...] = (
        'label',
        'area',
        'solidity',
        'eccentricity',
    ),
) -> pd.DataFrame:
    """Create a DataFrame containing region properties from image file.

    Args:
        image_path (Path): Path to the image file.
        band (int, optional): Band index to process. Defaults to 0.
        connectivity (int, optional): Connectivity for labeling. Defaults to 1.
        properties (tuple[str], optional): Properties to extract.
            Defaults to ('label', 'area', 'solidity', 'eccentricity').

    Returns:
        pd.DataFrame: DataFrame containing region properties.

    """
    image_array = skimage.io.imread(image_path)
    band_array = image_array[:, :, band]
    labels = label(band_array, connectivity=connectivity)
    df = pd.DataFrame(regionprops_table(labels, properties=properties))
    df['area_log10'] = np.log10(df['area'])
    df['image_name'] = image_path.stem
    return df


def create_regionprops_dataframe(
    image_paths: list[Path],
    band: int = 0,
    connectivity: int = 1,
    properties: tuple[Properties, ...] = (
        'label',
        'area',
        'solidity',
        'eccentricity',
    ),
) -> pd.DataFrame:
    """Create a DataFrame containing region properties from multiple images.

    Processes multiple image files and combines their region properties into
    a single DataFrame. Each image's regions are analyzed and additional columns
    for area_log10 and image_name are added.

    Args:
        image_paths (list[Path]): List of paths to image files.
        band (int, optional): Band index to process. Defaults to 0.
        connectivity (int, optional): Connectivity for labeling (1 or 2).
            Defaults to 1.
        properties (tuple[Properties, ...], optional): Properties to extract
            from regions. Defaults to ('label', 'area', 'solidity', 'eccentricity').

    Returns:
        pd.DataFrame: Combined DataFrame containing region properties from all
            images with columns for the specified properties plus 'area_log10'
            and 'image_name'.

    """
    dfs = []
    for image_path in image_paths:
        df = _create_regionprops_dataframe(
            image_path,
            band=band,
            connectivity=connectivity,
            properties=properties,
        )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False, axis=0)


def plot_kde_by_image(
    df: pd.DataFrame,
    column: str = 'area_log10',
    bw_adjust: float = 0.5,
    n_limit: int = 25,
):
    """Plot kernel density estimate (KDE) distributions by image.

    Creates a faceted plot showing KDE and rug plots for each image's
    distribution of the specified column. Titles include image names and
    sample counts.

    Args:
        df (pd.DataFrame): DataFrame containing region properties with
            'image_name' column and the specified column for plotting.
        column (str, optional): Column name to plot distribution for.
            Defaults to 'area_log10'.
        bw_adjust (float, optional): Bandwidth adjustment for KDE.
            Lower values show more detail. Defaults to 0.5.
        n_limit (int, optional): Maximum number of images to plot.
            Defaults to 25.

    """
    unique_images = df['image_name'].unique()[:n_limit]
    df_filtered = df[df['image_name'].isin(unique_images)]

    g = sns.FacetGrid(df_filtered, col='image_name', col_wrap=4, height=3)
    g.set_titles(col_template='{col_name}')
    g.map(sns.kdeplot, column, bw_adjust=bw_adjust)
    g.map(sns.rugplot, column)

    # Update titles with row counts
    counts = df_filtered.groupby('image_name').size()
    for ax in g.axes.flat:
        image_name = ax.get_title()
        if image_name and image_name in counts:
            count = counts[image_name]
            ax.set_title(f'{image_name}\n(n={count})')

    plt.tight_layout()
    plt.show()


def _apply_kmeans_with_ordered_labels(
    df_group: pd.DataFrame,
    column: str,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Apply k-means clustering and relabel based on mean area.

    Performs k-means clustering on the specified column and relabels clusters
    so that cluster 0 has the smallest mean area, cluster 1 has the next
    smallest, and so on.

    Args:
        df_group (pd.DataFrame): DataFrame or group to cluster.
        column (str): Column name to use for clustering.
        n_clusters (int): Number of clusters for k-means.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Array of cluster labels ordered by mean area
            (0 = smallest, n_clusters-1 = largest).

    """
    # Apply k-means clustering
    X = df_group[[column]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    initial_labels = kmeans.fit_predict(X)

    # Calculate mean area for each cluster
    df_temp = df_group.copy()
    df_temp['temp_cluster'] = initial_labels
    mean_areas = df_temp.groupby('temp_cluster')['area'].mean()

    # Create mapping: sort clusters by mean area (ascending)
    sorted_clusters = mean_areas.sort_values().index.tolist()
    label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(sorted_clusters)
    }

    # Apply mapping to relabel clusters
    relabeled = np.array([label_mapping[label] for label in initial_labels])

    return relabeled


def classify_cells_kmeans(
    df: pd.DataFrame,
    column: str = 'area_log10',
    random_state: int = 42,
    n_clusters: int = 2,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Classify cells into groups using k-means clustering.

    Uses k-means clustering to separate cells into groups, typically
    to distinguish between noise and foreground cells based on area.
    Can optionally perform clustering separately within each group.
    Cluster labels are standardized so that 0 represents smaller area
    and 1 represents larger area.

    Args:
        df (pd.DataFrame): DataFrame containing region properties with columns
            including the specified column for clustering and 'area' column.
        column (str, optional): Column name to use for clustering.
            Defaults to 'area_log10'.
        random_state (int, optional): Random state for reproducibility.
            Defaults to 42.
        n_clusters (int, optional): Number of clusters for k-means.
            Defaults to 2.
        group_by (str | None, optional): Column name to group by before
            clustering. If specified, k-means is applied separately within
            each group. Defaults to None.

    Returns:
        pd.DataFrame: Copy of input DataFrame with additional 'cluster' column
            containing cluster labels (0 for smaller area, 1 for larger area).

    """
    df_result = df.copy()

    if group_by is None:
        # Apply k-means clustering to entire dataset
        df_result['cluster'] = _apply_kmeans_with_ordered_labels(
            df_result, column, n_clusters, random_state
        )
    else:
        # Apply k-means clustering separately within each group
        cluster_labels = []
        for _, group in df_result.groupby(group_by):
            labels = _apply_kmeans_with_ordered_labels(
                group, column, n_clusters, random_state
            )
            cluster_labels.extend(labels)
        df_result['cluster'] = cluster_labels

    return df_result
