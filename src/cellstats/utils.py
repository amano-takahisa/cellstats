import math
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    region_props: list[skimage.measure._regionprops.RegionProperties],
    image: np.ndarray,
    n_buffer: int = 2,
    limit_num: int = 25,
):
    num = min(len(region_props), limit_num)
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
        array = image[slice_]
        axs[i].matshow(array)
        axs[i].set_title(
            f'Label: {region_props[i].label}\nArea: {int(region_props[i].area)}'
        )
    fig.tight_layout()
    plt.show()
