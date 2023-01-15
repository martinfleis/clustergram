# Installation

You can install clustergram from ``conda-forge`` with ``conda`` or ``mamba`` or from
PyPI with ``pip``:

```shell
mamba install clustergram -c conda-forge
```

or

```shell
pip install clustergram
```

This will install only the required dependencies (``pandas``, ``matplotlib``), but you
still need to install your selected backend (``scikit-learn`` and ``scipy`` or
``cuML``).

```shell
mamba install scikit-learn scipy -c conda-forge
```

For installation of ``cuML``, please refer to the [RAPIDS.AI
documentation](https://rapids.ai/start.html).

If you want to use interactive plotting of clustergrams, you will also need ``Bokeh``.

```shell
mamba install bokeh
```

## From source

If you prefer to use development version, you can install it from GitHub with pip or
clone and install a local copy.

```shell
pip install git+https://github.com/martinfleis/clustergram.git
```

Or clone and

```shell
cd clustergram
pip install .
```

## Development environment

If you want to have all the tools (except RAPIDS), you can use `environment-dev.yml`.

```sh
conda env create -f environment-dev.yml
```
