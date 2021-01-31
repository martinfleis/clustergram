"""
Clustergram - visualization and diagnostics for cluster analysis in Python

Copyright (C) 2020-2021  Martin Fleischmann

Clustergram is a Python implementation of R function written by Tal Galili.
https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/

Original idea is by Matthias Schonlau - http://www.schonlau.net/clustergram.html.

"""

from time import time
import pandas as pd
import numpy as np


class Clustergram:
    """
    Clustergram class mimicking the interface of clustering class (e.g. ``KMeans``).

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
        iterable of integer values to be tested as ``k``.
    backend : {'sklearn', 'cuML'} (default 'sklearn')
        Whether to use ``sklearn``'s implementation of KMeans and PCA or ``cuML``
        version. ``sklearn`` does computation on CPU, ``cuml`` on GPU.
    method : {'kmeans', 'gmm', 'minibatchkmeans'} (default 'kmeans')
        Clustering method. ``kmeans`` uses K-Means clustering, ``gmm``
        Gaussian Mixture Model, ``minibatchkmeans`` uses Mini Batch K-Means.
        ``gmm`` and ``minibatchkmeans`` are currently supported only
        with ``sklearn`` backend.
    verbose : bool (default True)
        Print progress and time of individual steps.
    **kwargs
        Additional arguments passed to the model (e.g. ``KMeans``),
        e.g. ``random_state``.


    Attributes
    ----------

    labels : DataFrame
        DataFrame with cluster labels for each iteration.
    cluster_centers : dict
        Dictionary with cluster centers for each iteration.


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
        self, k_range, backend="sklearn", method="kmeans", verbose=True, **kwargs,
    ):
        self.k_range = k_range

        # cleanup after API change
        kwargs.pop("pca_weighted", None)
        kwargs.pop("pca_kwargs", None)

        if backend not in ["sklearn", "cuML"]:
            raise ValueError(
                f'"{backend}" is not a supported backend. Use "sklearn" or "cuML".'
            )
        else:
            self.backend = backend

        supported = ["kmeans", "gmm", "minibatchkmeans"]
        if method not in supported:
            raise ValueError(
                f"'{method}' is not a supported method. "
                f"Only {supported} are supported now."
            )
        else:
            self.method = method

        self.engine_kwargs = kwargs
        self.verbose = verbose

        if self.backend == "sklearn":
            self.plot_data_pca = pd.DataFrame()
            self.plot_data = pd.DataFrame()
        else:
            import cudf

            self.plot_data_pca = cudf.DataFrame()
            self.plot_data = cudf.DataFrame()

    def __repr__(self):
        return (
            f"Clustergram(k_range={self.k_range}, backend='{self.backend}', "
            f"method='{self.method}', kwargs={self.engine_kwargs})"
        )

    def fit(self, data, **kwargs):
        """
        Compute (weighted) means of clusters.

        Parameters
        ----------
        data : array-like
            Input data to be clustered. It is expected that data are scaled. Can be
            ``numpy.array``, ``pandas.DataFrame`` or their RAPIDS counterparts.
        **kwargs
            Additional arguments passed to the ``.fit()`` method of the model,
            e.g. ``sample_weight``.

        Returns
        -------
        self
            Fitted clustergram.

        """
        self.data = data
        if self.backend == "sklearn":
            if self.method == "kmeans":
                self._kmeans_sklearn(data, minibatch=False, **kwargs)
            elif self.method == "minibatchkmeans":
                self._kmeans_sklearn(data, minibatch=True, **kwargs)
            elif self.method == "gmm":
                self.means = self._gmm_sklearn(data, **kwargs)
        if self.backend == "cuML":
            self.means = self._kmeans_cuml(data, **kwargs)

    def _kmeans_sklearn(self, data, minibatch, **kwargs):
        """Use scikit-learn KMeans"""
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans
        except ImportError:
            raise ImportError("scikit-learn is required to use `sklearn` backend.")

        self.labels = pd.DataFrame()
        self.cluster_centers = {}

        for n in self.k_range:
            s = time()
            if minibatch:
                results = MiniBatchKMeans(n_clusters=n, **self.engine_kwargs).fit(
                    data, **kwargs
                )
            else:
                results = KMeans(n_clusters=n, **self.engine_kwargs).fit(data, **kwargs)

            self.labels[n] = results.labels_
            self.cluster_centers[n] = results.cluster_centers_

            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None

    def _kmeans_cuml(self, data, **kwargs):
        """Use cuML KMeans"""
        try:
            from cuml import KMeans
            import cudf
        except ImportError:
            raise ImportError(
                "cuML, cuDF and cupy packages are required to use `cuML` backend."
            )

        self.labels = cudf.DataFrame()
        self.cluster_centers = {}

        for n in self.k_range:
            s = time()
            results = KMeans(n_clusters=n, **self.engine_kwargs).fit(data, **kwargs)
            self.labels[n] = results.labels_
            self.cluster_centers[n] = results.cluster_centers_

            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None

    def _gmm_sklearn(self, data, **kwargs):
        """Use sklearn.mixture.GaussianMixture"""
        try:
            from sklearn.mixture import GaussianMixture
            import numpy as np
            from scipy.stats import multivariate_normal
        except ImportError:
            raise ImportError(
                "scikit-learn and scipy are required to use `sklearn` "
                "backend and `gmm`."
            )

        if isinstance(data, pd.DataFrame):
            data = data.values

        self.labels = pd.DataFrame()
        self.cluster_centers = {}

        for n in self.k_range:
            s = time()
            results = GaussianMixture(n_components=n, **self.engine_kwargs).fit(
                data, **kwargs
            )
            centers = np.empty(shape=(results.n_components, data.shape[1]))
            for i in range(results.n_components):
                density = multivariate_normal(
                    cov=results.covariances_[i],
                    mean=results.means_[i],
                    allow_singular=True,
                ).logpdf(data)
                centers[i, :] = data[np.argmax(density)]

            self.labels[n] = results.predict(data)
            self.cluster_centers[n] = centers

            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None

    def silhouette_score(self, **kwargs):
        """
        Compute the mean Silhouette Coefficient of all samples.

        See the documentation of ``sklearn.metrics.silhouette_score`` for details.

        Once computed, resulting Series is available as ``Clustergram.silhouette``.

        Parameters
        ----------

        **kwargs
            Additional arguments passed to the silhouette_score function,
            e.g. ``sample_size``.

        Returns
        -------
        self.silhouette : pd.Series

        """
        from sklearn import metrics

        self.silhouette = pd.Series(name="silhouette_score")
        for k in self.k_range:
            if k > 1:
                self.silhouette.loc[k] = metrics.silhouette_score(
                    self.data, self.labels[k], **kwargs
                )
        return self.silhouette

    def calinski_harabasz_score(self):
        """
        Compute the Calinski and Harabasz score.

        See the documentation of ``sklearn.metrics.calinski_harabasz_score``
        for details.

        Once computed, resulting Series is available as
        ``Clustergram.calinski_harabasz``.

        Returns
        -------
        self.calinski_harabasz : pd.Series

        """
        from sklearn import metrics

        self.calinski_harabasz = pd.Series(name="calinski_harabasz_score")
        for k in self.k_range:
            if k > 1:
                self.calinski_harabasz.loc[k] = metrics.calinski_harabasz_score(
                    self.data, self.labels[k]
                )
        return self.calinski_harabasz

    def davies_bouldin_score(self):
        """
        Compute the Davies-Bouldin score.

        See the documentation of ``sklearn.metrics.davies_bouldin_score`` for details.

        Once computed, resulting Series is available as ``Clustergram.davies_bouldin``.

        Returns
        -------
        self.davies_bouldin : pd.Series

        """
        from sklearn import metrics

        self.davies_bouldin = pd.Series(name="davies_bouldin_score")
        for k in self.k_range:
            if k > 1:
                self.davies_bouldin.loc[k] = metrics.davies_bouldin_score(
                    self.data, self.labels[k]
                )
        return self.davies_bouldin

    def _compute_pca_means_sklearn(self, **pca_kwargs):
        """Compute PCA weighted cluster mean values using sklearn backend"""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=1, **pca_kwargs).fit(self.data).components_[0]

        for n in self.k_range:
            means = self.cluster_centers[n].dot(self.pca)
            self.plot_data_pca[n] = np.take(means, self.labels[n].values)

    def _compute_means_sklearn(self):
        """Compute cluster mean values using sklearn backend"""
        for n in self.k_range:
            means = np.mean(self.cluster_centers[n], axis=1)
            self.plot_data[n] = np.take(means, self.labels[n].values)

    def _compute_pca_means_cuml(self, **pca_kwargs):
        """Compute PCA weighted cluster mean values using cuML backend"""
        from cuml import PCA
        import cudf
        import cupy as cp

        self.pca = PCA(n_components=1, **pca_kwargs).fit(self.data)

        for n in self.k_range:
            if isinstance(self.data, cudf.DataFrame):
                means = self.cluster_centers[n].dot(self.pca.components_.values[0])
            else:
                means = self.cluster_centers[n].dot(self.pca.components_[0])
            self.plot_data_pca[n] = cp.take(means, self.labels[n].values)

    def _compute_means_cuml(self):
        """Compute cluster mean values using cuML backend"""
        import cupy as cp

        for n in self.k_range:
            means = self.cluster_centers[n].mean(axis=1)
            if isinstance(means, (cp.core.core.ndarray, np.ndarray)):
                self.plot_data[n] = means.take(self.labels[n].values)
            else:
                self.plot_data[n] = means.take(self.labels[n].values).to_array()

    def plot(
        self,
        ax=None,
        size=1,
        linewidth=1,
        cluster_style=None,
        line_style=None,
        figsize=None,
        k_range=None,
        pca_weighted=True,
        pca_kwargs={},
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
            Size of the resulting ``matplotlib.figure.Figure``. If the argument
            ``ax`` is given explicitly, ``figsize`` is ignored.
        k_range : iterable (default None)
            iterable of integer values to be plotted. In none, ``Clustergram.k_range``
            will be used. Has to be a subset of ``Clustergram.k_range``.
        pca_weighted : bool (default True)
            Whether use PCA weighted mean of clusters or standard mean of clusters on
            y-axis.
        pca_kwargs : dict (default {})
            Additional arguments passed to the PCA object,
            e.g. ``svd_solver``. Applies only if ``pca_weighted=True``.

        Returns
        -------
        ax : matplotlib axis instance
        """
        if pca_weighted:
            if self.plot_data_pca.empty:
                pca_kwargs.pop("n_components", None)

                if self.backend == "sklearn":
                    self._compute_pca_means_sklearn(**pca_kwargs)
                else:
                    self._compute_pca_means_cuml(**pca_kwargs)
        else:
            if self.plot_data.empty:
                if self.backend == "sklearn":
                    self._compute_means_sklearn()
                else:
                    self._compute_means_cuml()

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

        if pca_weighted:
            means = self.plot_data_pca
            ax.set_ylabel("PCA weighted mean of the clusters")
        else:
            means = self.plot_data
            ax.set_ylabel("Mean of the clusters")
        ax.set_xlabel("Number of clusters (k)")

        for i in k_range:
            cl = means[i].value_counts()

            if self.backend == "sklearn":
                ax.scatter(
                    [i] * i,
                    [cl.index],
                    cl * ((500 / len(means)) * size),
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
                    (cl * ((500 / len(means)) * size)).to_array(),
                    zorder=cl_zorder,
                    color=cl_c,
                    edgecolor=cl_ec,
                    linewidth=cl_lw,
                    **cluster_style,
                )

            try:
                if self.backend == "sklearn":
                    sub = means.groupby([i, i + 1]).count().reset_index()
                else:
                    sub = (
                        self.means.groupby([i, i + 1]).count().reset_index().to_pandas()
                    )
                for r in sub.itertuples():
                    ax.plot(
                        [i, i + 1],
                        [r[1], r[2]],
                        linewidth=r[3] * ((50 / len(means)) * linewidth),
                        color=l_c,
                        zorder=l_zorder,
                        solid_capstyle=solid_capstyle,
                        **line_style,
                    )
            except (KeyError, ValueError):
                pass
        return ax
