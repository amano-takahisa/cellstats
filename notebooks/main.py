import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import skimage
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd
    from cellstats import utils
    from skimage.measure import label, regionprops, regionprops_table
    import seaborn as sns
    return Path, utils


@app.cell
def _(utils):
    from importlib import reload

    reload(utils)
    return


@app.cell
def _(Path):
    data_dir = Path('data/input')
    file_name_pattern = '*.tif'
    return data_dir, file_name_pattern


@app.cell
def _(Path, data_dir, file_name_pattern):
    # check inputs
    def check_inputs(data_dir: Path, file_name_pattern: str):
        file_paths = list(data_dir.glob(file_name_pattern))
        if not data_dir.exists():
            raise FileNotFoundError(f'Data directory {data_dir} does not exist.')
        if len(file_paths) == 0:
            raise FileNotFoundError(
                f'No files found in {data_dir} matching {file_name_pattern}.'
            )


    check_inputs(data_dir=data_dir, file_name_pattern=file_name_pattern)
    return


@app.cell
def _(data_dir, file_name_pattern):
    image_paths = sorted(data_dir.glob(file_name_pattern))
    print(f'Found {len(image_paths)} images.')
    return (image_paths,)


@app.cell
def _(image_paths, utils):
    utils.plot_image(image_path=image_paths[18], band=0)
    return


@app.cell
def _(image_paths, utils):
    utils.plot_regions(image_path=image_paths[2], band=0, connectivity=1)
    return


@app.cell
def _(image_paths, utils):
    df = utils.create_regionprops_dataframe(
        image_paths=image_paths[0:20],
        band=0,
        connectivity=1,
        properties=('label', 'area', 'solidity', 'eccentricity'),
    )
    return (df,)


@app.cell
def _(df):
    df.sort_values(by=['image_name', 'area'])
    return


@app.cell
def _(df, utils):
    utils.plot_kde_by_image(df=df, column='area_log10', bw_adjust=0.5)
    return


@app.cell
def _(df, utils):
    classified_df = utils.classify_cells_kmeans(
        df=df,
        column='area_log10',
        random_state=42,
        n_classes=2,
        group_by='image_name',
    )
    classified_df
    return (classified_df,)


@app.cell
def _(Path, classified_df, image_paths, utils):
    input_image_paths = image_paths[0:20]
    outut_image_paths = [Path('data/output') / p.name for p in input_image_paths]
    utils.export_denoised_image(
        input_image_paths=input_image_paths,
        output_image_paths=outut_image_paths,
        class_df=classified_df,
        band=0,
        remove_classes=[0],
        class_column='class',
        connectivity=1,
        image_name_column='image_name',
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
