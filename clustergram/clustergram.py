"""
Clustergram - visualization and diagnostics for cluster analysis in Python

Copyright (C) 2020  Martin Fleischmann

Clustergram is a Python implementation of R function written by Tal Galili.
https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/

Original idea is by Matthias Schonlau - http://www.schonlau.net/clustergram.html.

"""


def _kmeans_sklearn(k_range, data, pca_weighted=True, **kwargs):
    """Use scikit-learn KMeans"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from pandas import DataFrame
        import numpy as np
    except ImportError:
        raise ImportError("scikit-learn package is required to use `sklearn` backend.")

    df = DataFrame()
    if pca_weighted:
        pca = PCA(1).fit(data)
    for n in k_range:
        results = KMeans(n_clusters=n, **kwargs).fit(data)
        cluster = results.labels_
        if pca_weighted:
            means = results.cluster_centers_.dot(pca.components_[0])
        else:
            means = np.mean(results.cluster_centers_, axis=1)
        df[n] = np.take(means, cluster)
    return df


def _kmeans_cuml(k_range, data, pca_weighted=True, **kwargs):
    """Use cuML KMeans"""
    try:
        from cuml import KMeans, PCA
        from cudf import DataFrame
        import cupy as cp
    except ImportError:
        raise ImportError("cuML package is required to use `cuML` backend.")

    df = DataFrame()
    if pca_weighted:
        pca = PCA(1).fit(data)
    for n in k_range:
        results = KMeans(n_clusters=n, **kwargs).fit(data)
        cluster = results.labels_
        if pca_weighted:
            if isinstance(results.cluster_centers_, DataFrame):
                means = results.cluster_centers_.values.dot(pca.components_.values[0])
            else:
                means = results.cluster_centers_.dot(pca.components_[0])
            df[n] = cp.take(means, cluster)
        else:
            means = results.cluster_centers_.mean(axis=1)
            df[n] = means.take(cluster).to_array()
    return df


def cluster_means(k_range, data, backend, pca_weighted=True, **kwargs):
    """
    Compute (weighted) means of clusters.

    Parameters
    ----------
    data : array-like
        Input data to be clustered. It is expected that data are scaled. Can be
        numpy.array, pandas.DataFrame or their RAPIDS counterparts.
    k_range : iterable
        iterable of integer values to be tested as k.
    pca_weighted : bool (default True)
        Whether use PCA weighted mean of clusters or standard mean of clusters.
    backend : string ('sklearn' or 'cuML', default 'sklearn')
        Whether to use `sklearn`'s implementation of KMeans and PCA or `cuML` version.
        Sklearn does computation on CPU, cuML on GPU.
    **kwargs
        Additional arguments passed to the KMeans object,
         e.g. ``random_state``.

    Returns
    -------
    df : DataFrame

    """

    if backend == "sklearn":
        return _kmeans_sklearn(k_range, data, pca_weighted=pca_weighted, **kwargs)
    if backend == "cuML":
        return _kmeans_cuml(k_range, data, pca_weighted=pca_weighted, **kwargs)
    raise ValueError(f'"{backend}" is not supported backend. Use "sklearn" or "cuML".')


def plot_clustergram(
    df,
    k_range,
    backend,
    ax=None,
    size=1,
    linewidth=0.1,
    cluster_style=None,
    line_style=None,
    pca_weighted=True,
):
    """
    Generate clustergram plot based on cluster centre mean values.

    Parameters
    ----------
    df : DataFrame
        DataFrame with (weighted) means of clusters.
    k_range : iterable
        iterable of integer values to be tested as k.
    backend : string ('sklearn' or 'cuML', default 'sklearn')
        Whether to use `sklearn`'s implementation of KMeans and PCA or `cuML` version.
        Sklearn does computation on CPU, cuML on GPU.
    ax : matplotlib.pyplot.Artist (default None)
        matplotlib axis on which to draw the plot
    size : float (default 1)
        multiplier of the size of a cluster centre indication. Size is determined as
        the count of observations in a cluster multiplied by ``size``.
    linewidth : float (default .1)
        multiplier of the linewidth of a branch. Line width is determined as
        the count of observations in a branch multiplied by `linewidth`.
    cluster_style : dict (default None)
        Style options to be passed on to the cluster centre plot, such
        as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
    line_style : dict (default None)
        Style options to be passed on to branches, such
        as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
    **kwargs
        Additional arguments passed to the KMeans object,
         e.g. ``random_state``.

    Returns
    -------
    ax : matplotlib axis instance
    """
    df["count"] = 1
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

    if cluster_style is None:
        cluster_style = {}
    cl_c = cluster_style.pop("color", "r")
    cl_ec = cluster_style.pop("edgecolor", "w")
    cl_lw = cluster_style.pop("linewidth", 2)
    cl_zorder = cluster_style.pop("zorder", 2)

    if line_style is None:
        line_style = {}
    l_c = line_style.pop("color", "k")
    l_zorder = line_style.pop("zorder", 1)
    solid_capstyle = line_style.pop("solid_capstyle", "butt")

    for i in k_range:
        cl = df[i].value_counts()

        if backend == "sklearn":
            ax.scatter(
                [i] * i,
                [cl.index],
                cl * size,
                zorder=cl_zorder,
                color=cl_c,
                edgecolor=cl_ec,
                linewidth=cl_lw,
                **cluster_style,
            )
        else:
            ax.scatter(
                [i] * i,
                cl.index.to_array(),
                (cl * size).to_array(),
                zorder=cl_zorder,
                color=cl_c,
                edgecolor=cl_ec,
                linewidth=cl_lw,
                **cluster_style,
            )

        try:
            if backend == "sklearn":
                sub = df.groupby([i, i + 1]).count().reset_index()[[i, i + 1, "count"]]
            else:
                sub = (
                    df.groupby([i, i + 1])
                    .count()
                    .reset_index()[[i, i + 1, "count"]]
                    .to_pandas()
                )
            for r in sub.itertuples():
                ax.plot(
                    [i, i + 1],
                    [r[1], r[2]],
                    linewidth=r[3] * linewidth,
                    color=l_c,
                    zorder=l_zorder,
                    solid_capstyle=solid_capstyle,
                    **line_style,
                )
        except (KeyError, ValueError):
            pass
    if pca_weighted:
        ax.set_ylabel("PCA weighted mean of the clusters")
    else:
        ax.set_ylabel("Mean of the clusters")
    ax.set_xlabel("Number of clusters (k)")
    return ax


def clustergram(
    data,
    k_range,
    pca_weighted=True,
    backend="sklearn",
    ax=None,
    size=1,
    linewidth=0.1,
    cluster_style=None,
    line_style=None,
    **kwargs,
):
    """
    Plot a clustergram.

    TODO: Long description

    Parameters
    ----------
    data : array-like
        Input data to be clustered. It is expected that data are scaled. Can be
        numpy.array, pandas.DataFrame or their RAPIDS counterparts.
    k_range : iterable
        iterable of integer values to be tested as k.
    pca_weighted : bool (default True)
        Whether use PCA weighted mean of clusters or standard mean of clusters.
    backend : string ('sklearn' or 'cuML', default 'sklearn')
        Whether to use `sklearn`'s implementation of KMeans and PCA or `cuML` version.
        Sklearn does computation on CPU, cuML on GPU.
    ax : matplotlib.pyplot.Artist (default None)
        matplotlib axis on which to draw the plot
    size : float (default 1)
        multiplier of the size of a cluster centre indication. Size is determined as
        the count of observations in a cluster multiplied by ``size``.
    linewidth : float (default .1)
        multiplier of the linewidth of a branch. Line width is determined as
        the count of observations in a branch multiplied by `linewidth`.
    cluster_style : dict (default None)
        Style options to be passed on to the cluster centre plot, such
        as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
    line_style : dict (default None)
        Style options to be passed on to branches, such
        as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
    **kwargs
        Additional arguments passed to the KMeans object,
         e.g. ``random_state``.

    Examples
    --------

    >>> clustergram(data, range(1, 9))

    Returns
    -------
    ax : matplotlib axis instance

    """
    clg = cluster_means(
        k_range, data, pca_weighted=pca_weighted, backend=backend, **kwargs
    )
    return plot_clustergram(
        clg,
        k_range,
        backend,
        ax=ax,
        pca_weighted=pca_weighted,
        size=size,
        linewidth=linewidth,
        cluster_style=cluster_style,
        line_style=line_style,
    )
