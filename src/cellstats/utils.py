import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure


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
