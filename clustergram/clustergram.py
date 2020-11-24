"""
Clustergram - visualization and diagnostics for cluster analysis in Python

Copyright (C) 2020  Martin Fleischmann

Clustergram is a Python implementation of R function written by Tal Galili.
https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/

Original idea is by Matthias Schonlau - http://www.schonlau.net/clustergram.html.

"""

from time import time


class Clustergram:
    """
    Clustergram class mimicking the interface of clustering class (e.g. KMeans).

    Clustergram is a graph used to examine how cluster members are assigned to clusters
    as the number of clusters increases. This graph is useful in
    exploratory analysis for nonhierarchical clustering algorithms such
    as k-means and for hierarchical cluster algorithms when the number of
    observations is large enough to make dendrograms impractical.

    Clustergram offers two backends for the computation - ``scikit-learn``
    which uses CPU and RAPIDS.AI ``cuML``, which uses GPU. Note that both
    are optional dependencies, but you will need at least one of them to
    generate clustergram.

    Parameters
    ----------
    k_range : iterable
        iterable of integer values to be tested as k.
    backend : string ('sklearn' or 'cuML', default 'sklearn')
        Whether to use `sklearn`'s implementation of KMeans and PCA or `cuML` version.
        Sklearn does computation on CPU, cuML on GPU.
    method : string ('kmeans' or 'gmm')
        Clustering method. ``kmeans`` uses KMeans clustering, 'gmm' Gaussian Mixture Model.
        'gmm' is currently supported only with 'sklearn' backend.
    pca_weighted : bool (default True)
        Whether use PCA weighted mean of clusters or standard mean of clusters.
    pca_kwargs : dict (default {})
        Additional arguments passed to the PCA object,
        e.g. ``svd_solver``. Applies only if ``pca_weighted=True``.
    verbose : bool (default True)
        Print progress and time of individual steps.
    **kwargs
        Additional arguments passed to the KMeans object,
        e.g. ``random_state``.


    Attributes
    ----------

    means : DataFrame
        DataFrame with (weighted) means of clusters.

    Examples
    --------
    >>> c_gram = clustergram.Clustergram(range(1, 9))
    >>> c_gram.fit(data)
    >>> c_gram.plot()

    Specifying parameters:

    >>> c_gram2 = clustergram.Clustergram(
    ...     range(1, 9), backend="cuML", pca_weighted=False, random_state=0
    ... )
    >>> c_gram2.fit(cudf_data)
    >>> c_gram2.plot(figsize=(12, 12))

    References
    ----------
    The clustergram: A graph for visualizing hierarchical and nonhierarchical
    cluster analyses: https://journals.sagepub.com/doi/10.1177/1536867X0200200405

    Tal Galili's R implementation:
    https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/


    """

    def __init__(
        self,
        k_range,
        backend="sklearn",
        method="kmeans",
        pca_weighted=True,
        pca_kwargs={},
        verbose=True,
        **kwargs,
    ):
        self.k_range = k_range

        if backend not in ["sklearn", "cuML"]:
            raise ValueError(
                f'"{backend}" is not a supported backend. Use "sklearn" or "cuML".'
            )
        else:
            self.backend = backend

        supported = ["kmeans", "gmm"]
        if method not in supported:
            raise ValueError(
                f'"{method}" is not a supported method. Only {supported} are supported now.'
            )
        else:
            self.method = method

        self.pca_weighted = pca_weighted
        self.kwargs = kwargs
        self.engine_kwargs = kwargs
        self.pca_kwargs = pca_kwargs
        self.verbose = verbose

    def fit(self, data, **kwargs):
        """
        Compute (weighted) means of clusters.

        Parameters
        ----------
        data : array-like
            Input data to be clustered. It is expected that data are scaled. Can be
            numpy.array, pandas.DataFrame or their RAPIDS counterparts.
        **kwargs
            Additional arguments passed to the KMeans.fit(),
            e.g. ``sample_weight``.

        Returns
        -------
        self
            Fitted clustergram.

        """
        if self.backend == "sklearn":
            if self.method == "kmeans":
                self.means = self._kmeans_sklearn(data, **kwargs)
            elif self.method == "gmm":
                self.means = self._gmm_sklearn(data, **kwargs)
        if self.backend == "cuML":
            self.means = self._kmeans_cuml(data, **kwargs)

    def _kmeans_sklearn(self, data, **kwargs):
        """Use scikit-learn KMeans"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from pandas import DataFrame
            import numpy as np
        except ImportError:
            raise ImportError(
                "scikit-learn, pandas and numpy are required to use `sklearn` backend."
            )

        df = DataFrame()
        if self.pca_weighted:
            s = time()
            self.pca_kwargs.pop("n_components", 1)
            pca = PCA(n_components=1, **self.pca_kwargs).fit(data)
            print(f"PCA computed in {time() - s} seconds.") if self.verbose else None

        for n in self.k_range:
            s = time()
            results = KMeans(n_clusters=n, **self.engine_kwargs).fit(data, **kwargs)
            cluster = results.labels_
            if self.pca_weighted:
                means = results.cluster_centers_.dot(pca.components_[0])
            else:
                means = np.mean(results.cluster_centers_, axis=1)
            df[n] = np.take(means, cluster)
            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None
        return df

    def _kmeans_cuml(self, data, **kwargs):
        """Use cuML KMeans"""
        try:
            from cuml import KMeans, PCA
            from cudf import DataFrame
            import cupy as cp
            import numpy as np
        except ImportError:
            raise ImportError(
                "cuML, cuDF and cupy packages are required to use `cuML` backend."
            )

        df = DataFrame()
        if self.pca_weighted:
            s = time()
            self.pca_kwargs.pop("n_components", 1)
            pca = PCA(n_components=1, **self.pca_kwargs).fit(data)
            print(f"PCA computed in {time() - s} seconds.") if self.verbose else None

        for n in self.k_range:
            s = time()
            results = KMeans(n_clusters=n, **self.engine_kwargs).fit(data, **kwargs)
            cluster = results.labels_
            if self.pca_weighted:
                if isinstance(results.cluster_centers_, DataFrame):
                    means = results.cluster_centers_.values.dot(
                        pca.components_.values[0]
                    )
                else:
                    means = results.cluster_centers_.dot(pca.components_[0])
                df[n] = cp.take(means, cluster)
            else:
                means = results.cluster_centers_.mean(axis=1)
                if isinstance(means, (cp.core.core.ndarray, np.ndarray)):
                    df[n] = means.take(cluster)
                else:
                    df[n] = means.take(cluster).to_array()
            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None
        return df

    def _gmm_sklearn(self, data, **kwargs):
        """Use sklearn.mixture.GaussianMixture"""
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.decomposition import PCA
            from pandas import DataFrame
            import numpy as np
            from scipy.stats import multivariate_normal
        except ImportError:
            raise ImportError(
                "scikit-learn, pandas and numpy are required to use `sklearn` backend."
            )

        if isinstance(data, DataFrame):
            data = data.values

        df = DataFrame()

        if self.pca_weighted:
            s = time()
            self.pca_kwargs.pop("n_components", 1)
            pca = PCA(n_components=1, **self.pca_kwargs).fit(data)
            print(f"PCA computed in {time() - s} seconds.") if self.verbose else None

        for n in self.k_range:
            s = time()
            results = GaussianMixture(n_components=n, **self.engine_kwargs).fit(
                data, **kwargs
            )
            cluster = results.predict(data)
            centers = np.empty(shape=(results.n_components, data.shape[1]))
            for i in range(results.n_components):
                density = multivariate_normal(
                    cov=results.covariances_[i], mean=results.means_[i]
                ).logpdf(data)
                centers[i, :] = data[np.argmax(density)]
            if self.pca_weighted:
                means = centers.dot(pca.components_[0])
            else:
                means = np.mean(centers, axis=1)
            df[n] = np.take(means, cluster)
            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None
        return df

    def plot(
        self,
        ax=None,
        size=1,
        linewidth=1,
        cluster_style=None,
        line_style=None,
        figsize=None,
        k_range=None,
    ):
        """
        Generate clustergram plot based on cluster centre mean values.

        Parameters
        ----------
        ax : matplotlib.pyplot.Artist (default None)
            matplotlib axis on which to draw the plot
        size : float (default 1)
            multiplier of the size of a cluster centre indication. Size is determined as
            ``500 / count`` of observations in a cluster multiplied by ``size``.
        linewidth : float (default 1)
            multiplier of the linewidth of a branch. Line width is determined as
            ``50 / count`` of observations in a branch multiplied by `linewidth`.
        cluster_style : dict (default None)
            Style options to be passed on to the cluster centre plot, such
            as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
        line_style : dict (default None)
            Style options to be passed on to branches, such
            as ``color``, ``linewidth``, ``edgecolor`` or ``alpha``.
        figsize : tuple of integers (default None)
            Size of the resulting matplotlib.figure.Figure. If the argument
            axes is given explicitly, figsize is ignored.
        k_range : iterable (default None)
            iterable of integer values to be plotted. In none, ``Clustergram.k_range``
            will be used. Has to be a substet of ``Clustergram.k_range``.

        Returns
        -------
        ax : matplotlib axis instance
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

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

        if k_range is None:
            k_range = self.k_range

        for i in k_range:
            cl = self.means[i].value_counts()

            if self.backend == "sklearn":
                ax.scatter(
                    [i] * i,
                    [cl.index],
                    cl * ((500 / len(self.means)) * size),
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
                    (cl * ((500 / len(self.means)) * size)).to_array(),
                    zorder=cl_zorder,
                    color=cl_c,
                    edgecolor=cl_ec,
                    linewidth=cl_lw,
                    **cluster_style,
                )

            try:
                if self.backend == "sklearn":
                    sub = self.means.groupby([i, i + 1]).count().reset_index()
                else:
                    sub = (
                        self.means.groupby([i, i + 1]).count().reset_index().to_pandas()
                    )
                for r in sub.itertuples():
                    ax.plot(
                        [i, i + 1],
                        [r[1], r[2]],
                        linewidth=r[3] * ((50 / len(self.means)) * linewidth),
                        color=l_c,
                        zorder=l_zorder,
                        solid_capstyle=solid_capstyle,
                        **line_style,
                    )
            except (KeyError, ValueError):
                pass
        if self.pca_weighted:
            ax.set_ylabel("PCA weighted mean of the clusters")
        else:
            ax.set_ylabel("Mean of the clusters")
        ax.set_xlabel("Number of clusters (k)")
        return ax
