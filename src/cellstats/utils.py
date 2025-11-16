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


def plot_image(image_path: Path, band: int = 0, band_color: str = 'red'):
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
    unique_images = df['image_name'].unique()[:n_limit]
    df_filtered = df[df['image_name'].isin(unique_images)]

    g = sns.FacetGrid(df_filtered, col='image_name', col_wrap=4, height=3)
    g.set_titles(col_template='{col_name}')
    g.map(sns.kdeplot, column, bw_adjust=bw_adjust)
    g.map(sns.rugplot, column)
    plt.tight_layout()
    plt.show()
