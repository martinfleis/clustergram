"""
Clustergram - visualization and diagnostics for cluster analysis in Python

Copyright (C) 2020-2021  Martin Fleischmann

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

    Clustergram offers three backends for the computation - ``scikit-learn`` and
    ``scipy`` which use CPU and RAPIDS.AI ``cuML``, which uses GPU. Note that all
    are optional dependencies but you will need at least one of them to
    generate clustergram.

    Alternatively, you can create clustergram using ``from_data`` or
    ``from_centers`` methods based on alternative clustering algorithms.

    Parameters
    ----------
    k_range : iterable (default None)
        iterable of integer values to be tested as ``k`` (number of cluster or
        components). Not required for hierarchical clustering but will be applied
        if given. It is recommended to always use limited range for hierarchical
        methods as unlimited clustergram can take a while to compute and for large
        number of observations is not legible.
    backend : {'sklearn', 'cuML', 'scipy'} (default None)
        Specify computational backend. Defaults to ``sklearn`` for ``'kmeans'``,
        ``'gmm'``, and ``'minibatchkmeans'`` methods and to ``'scipy'`` for any of
        hierarchical clustering methods. ``'scipy'`` uses ``sklearn`` for PCA
        computation if that is required.
        ``sklearn`` does computation on CPU, ``cuml`` on GPU.
    method : {'kmeans', 'gmm', 'minibatchkmeans', 'hierarchical'} (default 'kmeans')
        Clustering method.

        * ``kmeans`` uses K-Means clustering, either as ``sklearn.cluster.KMeans``
          or ``cuml.KMeans``.
        * ``gmm`` uses Gaussian Mixture Model as ``sklearn.mixture.GaussianMixture``
        * ``minibatchkmeans`` uses Mini Batch K-Means as
          ``sklearn.cluster.MiniBatchKMeans``
        * ``hierarchical`` uses hierarchical/agglomerative clustering as
          ``scipy.cluster.hierarchy.linkage``. See

        Note that ``gmm`` and ``minibatchkmeans`` are currently supported only
        with ``sklearn`` backend.
    verbose : bool (default True)
        Print progress and time of individual steps.
    **kwargs
        Additional arguments passed to the model (e.g. ``KMeans``),
        e.g. ``random_state``. Pass ``linkage`` to specify linkage method in
        case of hierarchical clustering (e.g. ``linkage='ward'``). See the
        documentation of scipy for details.


    Attributes
    ----------

    labels : DataFrame
        DataFrame with cluster labels for each iteration.
    cluster_centers : dict
        Dictionary with cluster centers for each iteration.
    linkage : scipy.cluster.hierarchy.linkage
        Linkage object for hierarchical methods.


    Examples
    --------
    >>> c_gram = clustergram.Clustergram(range(1, 9))
    >>> c_gram.fit(data)
    >>> c_gram.plot()

    Specifying parameters:

    >>> c_gram2 = clustergram.Clustergram(
    ...     range(1, 9), backend="cuML", random_state=0
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
        k_range=None,
        backend=None,
        method="kmeans",
        verbose=True,
        **kwargs,
    ):
        self.k_range = k_range

        # cleanup after API change
        kwargs.pop("pca_weighted", None)
        kwargs.pop("pca_kwargs", None)

        if backend is None:
            backend = "scipy" if method == "hierarchical" else "sklearn"

        allowed_backends = ["sklearn", "cuML", "scipy"]
        if backend not in allowed_backends:
            raise ValueError(
                f'"{backend}" is not a supported backend. '
                f"Use one of {allowed_backends}."
            )
        else:
            self.backend = backend

        supported = ["kmeans", "gmm", "minibatchkmeans", "hierarchical"]
        class_methods = ["from_centers", "from_data"]
        if method not in supported and method not in class_methods:
            raise ValueError(
                f"'{method}' is not a supported method. "
                f"Only {supported} are supported now."
            )
        else:
            self.method = method

        if self.k_range is None and self.method != "hierarchical":
            raise ValueError(f"'k_range' is mandatory for '{self.method}' method.")

        if (
            (self.backend == "cuML" and self.method != "kmeans")
            or (self.backend == "scipy" and self.method != "hierarchical")
            or (self.backend == "sklearn" and self.method == "hierarchical")
        ):
            raise ValueError(
                f"'{self.method}' method is not implemented "
                f"for '{self.backend}' backend. Use supported combination."
            )

        self.engine_kwargs = kwargs
        self.verbose = verbose

        if self.backend in ["sklearn", "scipy"]:
            self.plot_data_pca = pd.DataFrame()
            self.plot_data = pd.DataFrame()
        else:
            try:
                import cudf
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "cuML, cuDF and cupy packages are required to use `cuML` backend."
                )

            self.plot_data_pca = cudf.DataFrame()
            self.plot_data = cudf.DataFrame()

    def __repr__(self):
        return (
            f"Clustergram(k_range={self.k_range}, backend='{self.backend}', "
            f"method='{self.method}', kwargs={self.engine_kwargs})"
        )

    def fit(self, data, **kwargs):
        """
        Compute clustering for each k within set range.

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

        Examples
        --------
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)

        """
        self.data = data
        if self.backend == "sklearn":
            if self.method == "kmeans":
                self._kmeans_sklearn(data, minibatch=False, **kwargs)
            elif self.method == "minibatchkmeans":
                self._kmeans_sklearn(data, minibatch=True, **kwargs)
            elif self.method == "gmm":
                self._gmm_sklearn(data, **kwargs)
        if self.backend == "cuML":
            self._kmeans_cuml(data, **kwargs)
        if self.backend == "scipy":
            self._scipy_hierarchical(data, **kwargs)

    def _kmeans_sklearn(self, data, minibatch, **kwargs):
        """Use scikit-learn KMeans"""
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans
        except ImportError:
            raise ImportError("scikit-learn is required to use `sklearn` backend.")

        self.labels = pd.DataFrame()
        self.cluster_centers = {}

        for n in self.k_range:

            if n == 1:
                self.labels[n] = [0] * len(data)
                self.cluster_centers[n] = np.array([data.mean(axis=0)])

                print(
                    f"K={n} skipped. Mean computed from data directly."
                ) if self.verbose else None

                continue

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
            import cupy as cp
        except ImportError:
            raise ImportError(
                "cuML, cuDF and cupy packages are required to use `cuML` backend."
            )

        self.labels = cudf.DataFrame()
        self.cluster_centers = {}

        for n in self.k_range:

            if n == 1:
                self.labels[n] = [0] * len(data)
                if isinstance(data, cudf.DataFrame):
                    self.cluster_centers[n] = cudf.DataFrame(data.mean(axis=0)).T
                elif isinstance(data, cp.ndarray):
                    self.cluster_centers[n] = cp.array([data.mean(axis=0)])
                else:
                    self.cluster_centers[n] = np.array([data.mean(axis=0)])

                print(
                    f"K={n} skipped. Mean computed from data directly."
                ) if self.verbose else None

                continue

            s = time()
            results = KMeans(n_clusters=n, **self.engine_kwargs).fit(data, **kwargs)
            self.labels[n] = results.labels_
            self.cluster_centers[n] = results.cluster_centers_

            print(f"K={n} fitted in {time() - s} seconds.") if self.verbose else None

    def _gmm_sklearn(self, data, **kwargs):
        """Use sklearn.mixture.GaussianMixture"""
        try:
            from sklearn.mixture import GaussianMixture
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

    def _scipy_hierarchical(self, data, **kwargs):
        """Use scipy.cluster.hierarchy.linkage"""
        try:
            from scipy.cluster import hierarchy
        except ImportError:
            raise ImportError("scipy is required to use `scipy` backend.")

        method = self.engine_kwargs.pop("linkage", "single")
        self.linkage = hierarchy.linkage(data, method=method, **self.engine_kwargs)
        rootnode, nodelist = hierarchy.to_tree(self.linkage, rd=True)
        distances = [node.dist for node in nodelist if node.dist > 0][::-1]

        self.labels = pd.DataFrame()
        self.cluster_centers = {}

        if self.k_range is None:
            self.k_range = range(1, len(distances) + 1)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for i in self.k_range:
            d = distances[i - 1]
            lab = hierarchy.fcluster(self.linkage, d, criterion="distance")
            self.labels[i] = lab - 1
            self.cluster_centers[i] = data.groupby(lab).mean().values

    @classmethod
    def from_centers(cls, cluster_centers, labels, data=None):
        """Create clustergram based on cluster centers dictionary and labels DataFrame

        Parameters
        ----------

        cluster_centers : dict
            dictionary of cluster centers with keys encoding the number of clusters
            and values being ``M``x````N`` arrays where ``M`` == key and ``N`` ==
            number of variables in the original dataset.
            Entries should be ordered based on keys.
        labels : pandas.DataFrame
            DataFrame with columns representing cluster labels and rows representing
            observations. Columns must be equal to ``cluster_centers`` keys.
        data : array-like (optional)
            array used as an input of the clustering algorithm with ``N`` columns.
            Required for `plot(pca_weighted=True)` plotting option. Otherwise only
            `plot(pca_weighted=False)` is available.

        Returns
        -------
        clustegram.Clustergram

        Notes
        -----
        The algortihm uses ``sklearn`` and ``pandas`` to generate clustergram.
        GPU option is not implemented.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
        >>> labels
           1  2  3
        0  0  0  0
        1  0  0  2
        2  0  1  1
        >>> centers = {
        ...             1: np.array([[0, 0]]),
        ...             2: np.array([[-1, -1], [1, 1]]),
        ...             3: np.array([[-1, -1], [1, 1], [0, 0]]),
        ...         }
        >>> cgram = Clustergram.from_centers(centers, labels)
        >>> cgram.plot(pca_weighted=False)

        >>> data = np.array([[-1, -1], [1, 1], [0, 0]])
        >>> cgram = Clustergram.from_centers(centers, labels, data=data)
        >>> cgram.plot()

        """
        if not (list(cluster_centers.keys()) == labels.columns).all():
            raise ValueError("'cluster_centers' keys do not match 'labels' columns.")

        cgram = cls(k_range=list(cluster_centers.keys()), method="from_centers")

        cgram.cluster_centers = cluster_centers
        cgram.labels = labels
        cgram.backend = "sklearn"

        if data is not None:
            cgram.data = data

        return cgram

    @classmethod
    def from_data(cls, data, labels, method="mean"):
        """Create clustergram based on data and labels DataFrame

        Cluster centers are created as mean values or median values as a
        groupby function over data using individual labels.

        Parameters
        ----------

        data : array-like
            array used as an input of the clustering algorithm in the ``(M, N)`` shape
            where ``M`` == number of observations and ``N`` == number of variables
        labels : pandas.DataFrame
            DataFrame with columns representing cluster labels and rows representing
            observations. Columns must be equal to ``cluster_centers`` keys.
        method : {'mean', 'median'}, default 'mean'
            Method of computation of cluster centres.

        Returns
        -------
        clustegram.Clustergram

        Notes
        -----
        The algortihm uses ``sklearn`` and ``pandas`` to generate clustergram.
        GPU option is not implemented.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = np.array([[-1, -1, 0, 10], [1, 1, 10, 2], [0, 0, 20, 4]])
        >>> data
        array([[-1, -1,  0, 10],
               [ 1,  1, 10,  2],
               [ 0,  0, 20,  4]])
        >>> labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
        >>> labels
           1  2  3
        0  0  0  0
        1  0  0  2
        2  0  1  1
        >>> cgram = Clustergram.from_data(data, labels)
        >>> cgram.plot()

        """

        cgram = cls(k_range=list(labels.columns), method="from_data")

        cgram.cluster_centers = {}
        cgram.data = data

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for i in cgram.k_range:
            if method == "mean":
                cgram.cluster_centers[i] = data.groupby(labels[i].values).mean().values
            elif method == "median":
                cgram.cluster_centers[i] = (
                    data.groupby(labels[i].values).median().values
                )
            else:
                raise ValueError(
                    f"'{method}' is not supported. Use 'mean' or 'median'."
                )

        cgram.labels = labels
        cgram.backend = "sklearn"

        return cgram

    def silhouette_score(self, **kwargs):
        """
        Compute the mean Silhouette Coefficient of all samples.

        See the documentation of ``sklearn.metrics.silhouette_score`` for details.

        Once computed, resulting Series is available as ``Clustergram.silhouette``.
        Calling the original method will compute the score from the beginning.

        Parameters
        ----------

        **kwargs
            Additional arguments passed to the silhouette_score function,
            e.g. ``sample_size``.

        Returns
        -------
        silhouette : pd.Series

        Notes
        -----
        The algortihm uses ``sklearn``.
        With ``cuML`` backend, data are converted on the fly.

        Examples
        --------
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> c_gram.silhouette_score()
        2    0.702450
        3    0.644272
        4    0.767728
        5    0.948991
        6    0.769985
        7    0.575644
        Name: silhouette_score, dtype: float64

        Once computed:

        >>> c_gram.silhouette
        2    0.702450
        3    0.644272
        4    0.767728
        5    0.948991
        6    0.769985
        7    0.575644
        Name: silhouette_score, dtype: float64

        """
        from sklearn import metrics

        self.silhouette = pd.Series(name="silhouette_score", dtype="float64")

        if self.backend in ["sklearn", "scipy"]:
            for k in self.k_range:
                if k > 1:
                    self.silhouette.loc[k] = metrics.silhouette_score(
                        self.data, self.labels[k], **kwargs
                    )
        else:
            if hasattr(self.data, "to_pandas"):
                data = self.data.to_pandas()
            else:
                data = self.data.get()

            for k in self.k_range:
                if k > 1:
                    self.silhouette.loc[k] = metrics.silhouette_score(
                        data, self.labels[k].to_pandas(), **kwargs
                    )

        return self.silhouette

    def calinski_harabasz_score(self):
        """
        Compute the Calinski and Harabasz score.

        See the documentation of ``sklearn.metrics.calinski_harabasz_score``
        for details.

        Once computed, resulting Series is available as
        ``Clustergram.calinski_harabasz``. Calling the original method will
        compute the score from the beginning.

        Returns
        -------
        calinski_harabasz : pd.Series

        Notes
        -----
        The algortihm uses ``sklearn``.
        With ``cuML`` backend, data are converted on the fly.

        Examples
        --------
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> c_gram.calinski_harabasz_score()
        2      23.176629
        3      30.643018
        4      55.223336
        5    3116.435184
        6    3899.068689
        7    4439.306049
        Name: calinski_harabasz_score, dtype: float64

        Once computed:

        >>> c_gram.calinski_harabasz
        2      23.176629
        3      30.643018
        4      55.223336
        5    3116.435184
        6    3899.068689
        7    4439.306049
        Name: calinski_harabasz_score, dtype: float64

        """
        from sklearn import metrics

        self.calinski_harabasz = pd.Series(
            name="calinski_harabasz_score", dtype="float64"
        )

        if self.backend in ["sklearn", "scipy"]:
            for k in self.k_range:
                if k > 1:
                    self.calinski_harabasz.loc[k] = metrics.calinski_harabasz_score(
                        self.data, self.labels[k]
                    )
        else:
            if hasattr(self.data, "to_pandas"):
                data = self.data.to_pandas()
            else:
                data = self.data.get()

            for k in self.k_range:
                if k > 1:
                    self.calinski_harabasz.loc[k] = metrics.calinski_harabasz_score(
                        data, self.labels[k].to_pandas()
                    )
        return self.calinski_harabasz

    def davies_bouldin_score(self):
        """
        Compute the Davies-Bouldin score.

        See the documentation of ``sklearn.metrics.davies_bouldin_score`` for details.

        Once computed, resulting Series is available as ``Clustergram.davies_bouldin``.
        Calling the original method will recompute the score.

        Returns
        -------
        davies_bouldin : pd.Series

        Notes
        -----
        The algortihm uses ``sklearn``.
        With ``cuML`` backend, data are converted on the fly.

        Examples
        --------
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> c_gram.davies_bouldin_score()
        2    0.249366
        3    0.351812
        4    0.347580
        5    0.055679
        6    0.030516
        7    0.025207
        Name: davies_bouldin_score, dtype: float64

        Once computed:

        >>> c_gram.davies_bouldin
        2    0.249366
        3    0.351812
        4    0.347580
        5    0.055679
        6    0.030516
        7    0.025207
        Name: davies_bouldin_score, dtype: float64

        """
        from sklearn import metrics

        self.davies_bouldin = pd.Series(name="davies_bouldin_score", dtype="float64")

        if self.backend in ["sklearn", "scipy"]:
            for k in self.k_range:
                if k > 1:
                    self.davies_bouldin.loc[k] = metrics.davies_bouldin_score(
                        self.data, self.labels[k]
                    )
        else:
            if hasattr(self.data, "to_pandas"):
                data = self.data.to_pandas()
            else:
                data = self.data.get()

            for k in self.k_range:
                if k > 1:
                    self.davies_bouldin.loc[k] = metrics.davies_bouldin_score(
                        data, self.labels[k].to_pandas()
                    )

        return self.davies_bouldin

    def _compute_pca_means_sklearn(self, **pca_kwargs):
        """Compute PCA weighted cluster mean values using sklearn backend"""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=1, **pca_kwargs).fit(self.data).components_[0]
        self.link_pca = {}

        for n in self.k_range:
            means = self.cluster_centers[n].dot(self.pca)
            self.plot_data_pca[n] = np.take(means, self.labels[n].values)
            self.link_pca[n] = dict(zip(means, range(n)))

    def _compute_means_sklearn(self):
        """Compute cluster mean values using sklearn backend"""
        self.link = {}

        for n in self.k_range:
            means = np.mean(self.cluster_centers[n], axis=1)
            self.plot_data[n] = np.take(means, self.labels[n].values)
            self.link[n] = dict(zip(means, range(n)))

    def _compute_pca_means_cuml(self, **pca_kwargs):
        """Compute PCA weighted cluster mean values using cuML backend"""
        from cuml import PCA
        import cudf
        import cupy as cp

        self.pca = PCA(n_components=1, **pca_kwargs).fit(self.data)
        self.link_pca = {}

        for n in self.k_range:
            if isinstance(self.data, cudf.DataFrame):
                means = self.cluster_centers[n].values.dot(
                    self.pca.components_.values[0]
                )
            else:
                means = self.cluster_centers[n].dot(self.pca.components_[0])
            self.plot_data_pca[n] = cp.take(means, self.labels[n].values)
            self.link_pca[n] = dict(zip(means.tolist(), range(n)))

    def _compute_means_cuml(self):
        """Compute cluster mean values using cuML backend"""
        import cupy as cp

        self.link = {}

        for n in self.k_range:
            means = self.cluster_centers[n].mean(axis=1)
            if isinstance(means, (cp.core.core.ndarray, np.ndarray)):
                self.plot_data[n] = means.take(self.labels[n].values)
                self.link[n] = dict(zip(means.tolist(), range(n)))
            else:
                self.plot_data[n] = means.take(self.labels[n].values).to_array()
                self.link[n] = dict(zip(means.values.tolist(), range(n)))

    def _compute_means(self, pca_weighted, pca_kwargs):
        if pca_weighted:
            if self.plot_data_pca.empty:
                pca_kwargs.pop("n_components", None)

                if self.backend in ["sklearn", "scipy"]:
                    self._compute_pca_means_sklearn(**pca_kwargs)
                else:
                    self._compute_pca_means_cuml(**pca_kwargs)
        else:
            if self.plot_data.empty:
                if self.backend in ["sklearn", "scipy"]:
                    self._compute_means_sklearn()
                else:
                    self._compute_means_cuml()

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

        Examples
        --------
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> c_gram.plot()

        Notes
        -----
        Before plotting, ``Clustergram`` needs to compute the summary values.
        Those are computed on the first call of each option (pca_weighted=True/False).

        """
        self._compute_means(pca_weighted, pca_kwargs)

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

            if self.backend in ["sklearn", "scipy"]:
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
                if self.backend in ["sklearn", "scipy"]:
                    sub = means.groupby([i, i + 1]).count().reset_index()
                else:
                    sub = means.groupby([i, i + 1]).count().reset_index().to_pandas()
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

    def bokeh(
        self,
        fig=None,
        size=1,
        line_width=1,
        cluster_style=None,
        line_style=None,
        figsize=None,
        pca_weighted=True,
        pca_kwargs={},
    ):
        """
        Generate interactive clustergram plot based on cluster centre mean values using
        Bokeh.

        Requires ``bokeh``.

        Parameters
        ----------
        fig : bokeh.plotting.figure.Figure (default None)
            bokeh figure on which to draw the plot
        size : float (default 1)
            multiplier of the size of a cluster centre indication. Size is determined as
            ``50 / count`` of observations in a cluster multiplied by ``size``.
        line_width : float (default 1)
            multiplier of the linewidth of a branch. Line width is determined as
            ``50 / count`` of observations in a branch multiplied by `line_width`.
        cluster_style : dict (default None)
            Style options to be passed on to the cluster centre plot, such
            as ``color``, ``line_width``, ``line_color`` or ``alpha``.
        line_style : dict (default None)
            Style options to be passed on to branches, such
            as ``color``, ``line_width``, ``line_color`` or ``alpha``.
        figsize : tuple of integers (default None)
            Size of the resulting ``bokeh.plotting.figure.Figure``. If the argument
            ``figure`` is given explicitly, ``figsize`` is ignored.
        pca_weighted : bool (default True)
            Whether use PCA weighted mean of clusters or standard mean of clusters on
            y-axis.
        pca_kwargs : dict (default {})
            Additional arguments passed to the PCA object,
            e.g. ``svd_solver``. Applies only if ``pca_weighted=True``.

        Returns
        -------
        figure : bokeh figure instance

        Examples
        --------
        >>> from bokeh.plotting import show
        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> f = c_gram.bokeh()
        >>> show(f)

        For the best experience in Jupyter notebooks, specify bokeh output first:

        >>> from bokeh.io import output_notebook
        >>> from bokeh.plotting import show
        >>> output_notebook()

        >>> c_gram = clustergram.Clustergram(range(1, 9))
        >>> c_gram.fit(data)
        >>> f = c_gram.bokeh()
        >>> show(f)

        Notes
        -----
        Before plotting, ``Clustergram`` needs to compute the summary values.
        Those are computed on the first call of each option (pca_weighted=True/False).

        """
        try:
            from bokeh.plotting import figure, ColumnDataSource
            from bokeh.models import HoverTool
        except ImportError:
            raise ImportError("'bokeh' is required to use bokeh plotting backend.")

        self._compute_means(pca_weighted, pca_kwargs)

        if pca_weighted:
            means = self.plot_data_pca
            links = self.link_pca
            ylabel = "PCA weighted mean of the clusters"
        else:
            means = self.plot_data
            links = self.link
            ylabel = "Mean of the clusters"

        if fig is None:
            if figsize is None:
                figsize = (600, 500)
            fig = figure(
                plot_width=figsize[0],
                plot_height=figsize[1],
                x_axis_label="Number of clusters (k)",
                y_axis_label=ylabel,
            )

        if cluster_style is None:
            cluster_style = {}
        cl_c = cluster_style.pop("color", "red")
        cl_ec = cluster_style.pop("line_color", "white")
        cl_lw = cluster_style.pop("line_width", 2)

        if line_style is None:
            line_style = {}
        l_c = line_style.pop("color", "black")
        line_cap = line_style.pop("line_cap", "round")

        x = []
        y = []
        sizes = []
        count = []
        ratio = []
        cluster_labels = []

        total = len(means)
        for i in self.k_range:
            cl = means[i].value_counts()
            x += [i] * i
            y += cl.index.values.tolist()
            count += cl.values.tolist()
            ratio += ((cl / total) * 100).values.tolist()
            sizes += (cl * ((50 / len(means)) * size)).values.tolist()
            cluster_labels += [links[i][x] for x in cl.index.values.tolist()]

        source = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
                size=sizes,
                count=count,
                ratio=ratio,
                cluster_labels=cluster_labels,
            )
        )

        tooltips = [
            ("Number of observations", "@count (@ratio%)"),
            ("Cluster label", "@cluster_labels"),
        ]

        stop = max(self.k_range)
        for i in self.k_range:
            if i < stop:
                sub = means.groupby([i, i + 1]).count().reset_index()
                if self.backend == "cuML":
                    sub = sub.to_pandas()
                for r in sub.itertuples():
                    fig.line(
                        [i, i + 1],
                        [r[1], r[2]],
                        line_width=r[3] * ((50 / len(means)) * line_width),
                        line_cap=line_cap,
                        color=l_c,
                        **line_style,
                    )

        circle = fig.circle(
            "x",
            "y",
            size="size",
            source=source,
            color=cl_c,
            line_color=cl_ec,
            line_width=cl_lw,
            **cluster_style,
        )
        hover = HoverTool(tooltips=tooltips, renderers=[circle])
        fig.add_tools(hover)

        return fig
