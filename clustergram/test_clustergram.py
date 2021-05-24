from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import pytest
from bokeh.embed import json_item


try:
    import cudf
    import cuml
    import cupy as cp

    RAPIDS = True
except (ImportError, ModuleNotFoundError):
    RAPIDS = False

from clustergram import Clustergram

n_samples = 10
n_features = 2

n_clusters = 5
random_state = 0

device_data, device_labels = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=random_state,
    cluster_std=0.1,
)

data = pd.DataFrame(device_data)


def test_sklearn_kmeans():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.3448529748438847,
        2.793993337861089,
        2.8270525934022026,
        2.513999519667852,
        2.344852974843884,
        2.059790669470653,
        2.271614946625764,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.21726383145643727, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        2.3448529748438847, rel=1e-15
    )


def test_sklearn_minibatchkmeans():
    clustergram = Clustergram(
        range(1, 8),
        backend="sklearn",
        method="minibatchkmeans",
        random_state=random_state,
    )
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.3448529748438847,
        2.7950777425566464,
        2.8277733533088534,
        2.522309579003138,
        2.3452078554650075,
        2.0602593594306526,
        2.2717280475246584,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.2074208059962391, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        2.3469362205672746, rel=1e-15
    )


def test_sklearn_gmm():
    clustergram = Clustergram(
        range(1, 8), backend="sklearn", method="gmm", random_state=random_state
    )
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.6841643400794335,
        3.1705471246923214,
        2.52976769800815,
        2.74946247582996,
        2.333257525098582,
        2.0444680314214723,
        2.2697082687156915,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-6)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.5099395641282745, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        2.4439850629293924, rel=1e-15
    )


@pytest.mark.skipif(
    not RAPIDS,
    reason="RAPIDS not available.",
)
def test_cuml_kmeans():
    n_samples = 10
    n_features = 2

    n_clusters = 5
    random_state = 0

    device_data, device_labels = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    data = cudf.DataFrame(device_data)

    # cudf.DataFrame
    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        3.7674055099487305,
        2.7064273357391357,
        3.451129913330078,
        4.223802089691162,
        4.125243663787842,
        2.953890800476074,
        3.4818685054779053,
    ]
    assert expected == [
        pytest.approx(float(clustergram.cluster_centers[x].mean().mean()), rel=1e-6)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        1.1016593594032404, rel=1e-10
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        3.7674053507191796, rel=1e-10
    )

    # cupy array
    data = device_data

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        3.7674055099487305,
        2.7064273357391357,
        3.451129913330078,
        4.223802089691162,
        4.125243663787842,
        2.953890800476074,
        3.4818685054779053,
    ]
    assert expected == [
        pytest.approx(float(cp.mean(clustergram.cluster_centers[x])), rel=1e-6)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        1.1016593081610544, rel=1e-6
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        3.7674053737095425, rel=1e-6
    )


def test_hierarchical():
    clustergram = Clustergram(range(1, 8), method="hierarchical")
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.344852974843885,
        2.793993337861089,
        2.827052593402202,
        2.513999519667852,
        2.3448529748438847,
        2.059790669470653,
        2.2716149466257645,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.2172638314564372, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        2.3448529748438847, rel=1e-15
    )


def test_hierarchical_array():
    clustergram = Clustergram(method="hierarchical")
    clustergram.fit(data.values)

    for i in range(1, 10):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (10, 9)
    assert clustergram.labels.notna().all().all()


def test_errors():
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), backend="nonsense")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="nonsense")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="kmeans", backend="scipy")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="hieararchical", backend="sklearn")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="gmm", backend="cuML")
    with pytest.raises(ValueError):
        Clustergram()


def test_repr():
    expected = (
        "Clustergram(k_range=range(1, 30), backend='sklearn', "
        "method='kmeans', kwargs={'n_init': 10})"
    )
    clustergram = Clustergram(range(1, 30), n_init=10)
    assert expected == clustergram.__repr__()


def test_silhouette_score():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.silhouette_score(),
        pd.Series(
            [0.70244987, 0.64427202, 0.76772759, 0.94899084, 0.76998519, 0.57564372],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.silhouette,
        pd.Series(
            [0.70244987, 0.64427202, 0.76772759, 0.94899084, 0.76998519, 0.57564372],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
    )


@pytest.mark.skipif(
    not RAPIDS,
    reason="RAPIDS not available.",
)
def test_silhouette_score_cuml():
    n_samples = 10
    n_features = 2

    n_clusters = 5
    random_state = 0

    device_data, device_labels = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    data = cudf.DataFrame(device_data)

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.silhouette_score(),
        pd.Series(
            [
                0.7494349479675293,
                0.9806153178215027,
                0.6721830368041992,
                0.39418715238571167,
                0.44574037194252014,
                0.08033210784196854,
            ],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
    )

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(device_data)

    pd.testing.assert_series_equal(
        clustergram.silhouette_score(),
        pd.Series(
            [
                0.7494349479675293,
                0.9806153178215027,
                0.6721830368041992,
                0.39418715238571167,
                0.44574037194252014,
                0.08033210784196854,
            ],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
    )


def test_calinski_harabasz_score():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz_score(),
        pd.Series(
            [
                23.17662874,
                30.64301789,
                55.22333618,
                3116.43518408,
                3899.06868932,
                4439.30604863,
            ],
            index=list(range(2, 8)),
            name="calinski_harabasz_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz,
        pd.Series(
            [
                23.17662874,
                30.64301789,
                55.22333618,
                3116.43518408,
                3899.06868932,
                4439.30604863,
            ],
            index=list(range(2, 8)),
            name="calinski_harabasz_score",
        ),
    )


@pytest.mark.skipif(
    not RAPIDS,
    reason="RAPIDS not available.",
)
def test_calinski_harabasz_score_cuml():
    n_samples = 10
    n_features = 2

    n_clusters = 5
    random_state = 0

    device_data, device_labels = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    data = cudf.DataFrame(device_data)

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz_score(),
        pd.Series(
            [
                25.619150510634366,
                15374.042816067375,
                10813.16845006968,
                8818.1163716754,
                8070.657293970755,
                7259.89764652579,
            ],
            index=list(range(2, 8)),
            name="calinski_harabasz_score",
        ),
    )

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(device_data)

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz_score(),
        pd.Series(
            [
                25.619150510634366,
                15374.042816067375,
                10813.16845006968,
                8818.1163716754,
                8070.657293970755,
                7259.89764652579,
            ],
            index=list(range(2, 8)),
            name="calinski_harabasz_score",
        ),
    )


def test_davies_bouldin_score():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.davies_bouldin_score(),
        pd.Series(
            [0.2493657, 0.35181197, 0.34758021, 0.05567944, 0.03051626, 0.02520726],
            index=list(range(2, 8)),
            name="davies_bouldin_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.davies_bouldin,
        pd.Series(
            [0.2493657, 0.35181197, 0.34758021, 0.05567944, 0.03051626, 0.02520726],
            index=list(range(2, 8)),
            name="davies_bouldin_score",
        ),
    )


@pytest.mark.skipif(
    not RAPIDS,
    reason="RAPIDS not available.",
)
def test_davies_bouldin_score_cuml():
    n_samples = 10
    n_features = 2

    n_clusters = 5
    random_state = 0

    device_data, device_labels = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    data = cudf.DataFrame(device_data)

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.davies_bouldin_score(),
        pd.Series(
            [
                0.3107512701086121,
                0.02263161666570639,
                0.2261582258142144,
                0.3839688146565784,
                0.13388392354928222,
                0.279734367840293,
            ],
            index=list(range(2, 8)),
            name="davies_bouldin_score",
        ),
    )

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(device_data)

    pd.testing.assert_series_equal(
        clustergram.davies_bouldin_score(),
        pd.Series(
            [
                0.3107512701086121,
                0.02263161666570639,
                0.2261582258142144,
                0.3839688146565784,
                0.13388392354928222,
                0.279734367840293,
            ],
            index=list(range(2, 8)),
            name="davies_bouldin_score",
        ),
    )


def test_from_data_mean():
    data = np.array([[-1, -1, 0, 10], [1, 1, 10, 2], [0, 0, 20, 4]])
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    clustergram = Clustergram.from_data(data, labels)

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -7.820673888000655, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        3.8333333333333335, rel=1e-15
    )


def test_from_data_median():
    data = np.array([[-1, -1, 0, 10], [1, 1, 10, 2], [0, 0, 20, 4]])
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    clustergram = Clustergram.from_data(data, labels, method="median")

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -7.958519683972767, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        3.7222222222222228, rel=1e-15
    )


def test_from_data_nonsense():
    data = np.array([[-1, -1, 0, 10], [1, 1, 10, 2], [0, 0, 20, 4]])
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    with pytest.raises(ValueError, match="'nonsense' is not supported."):
        Clustergram.from_data(data, labels, method="nonsense")


def test_from_data_index():
    data = pd.DataFrame(
        np.array([[-1, -1, 0, 10], [1, 1, 10, 2], [0, 0, 20, 4]]), index=["a", "b", "c"]
    )
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    clustergram = Clustergram.from_data(data, labels)
    clustergram.plot()
    clustergram.plot(pca_weighted=False)

    clustergram = Clustergram.from_data(data, labels, method="median")
    clustergram.plot()
    clustergram.plot(pca_weighted=False)


def test_from_centers():
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    centers = {
        1: np.array([[0, 0]]),
        2: np.array([[-1, -1], [1, 1]]),
        3: np.array([[-1, -1], [1, 1], [0, 0]]),
    }
    clustergram = Clustergram.from_centers(centers, labels)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.mean().mean() == pytest.approx(
        -0.1111111111111111, rel=1e-15
    )

    labels = pd.DataFrame({2: [0, 0, 0], 3: [0, 0, 1], 4: [0, 2, 1]})
    centers = {
        1: np.array([[0, 0]]),
        2: np.array([[-1, -1], [1, 1]]),
        3: np.array([[-1, -1], [1, 1], [0, 0]]),
    }
    with pytest.raises(ValueError, match="'cluster_centers' keys do not match"):
        Clustergram.from_centers(centers, labels)


def test_from_centers_data():
    labels = pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 1], 3: [0, 2, 1]})
    centers = {
        1: np.array([[0, 0]]),
        2: np.array([[-1, -1], [1, 1]]),
        3: np.array([[-1, -1], [1, 1], [0, 0]]),
    }
    data = np.array([[-1, -1], [1, 1], [0, 0]])
    clustergram = Clustergram.from_centers(centers, labels, data)

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_weighted=True)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.15713484026367722, rel=1e-15
    )


def test_bokeh():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    f = clustergram.bokeh(pca_kwargs=dict(random_state=random_state))
    out = str(json_item(f, "clustergram"))

    assert out.count("data") == 56
    assert out.count("'x'") == 140
    assert out.count("'y'") == 140
    assert "cluster_labels" in out
    assert "count" in out
    assert "ratio" in out
    assert "size" in out

    f = clustergram.bokeh(pca_weighted=False)
    out = str(json_item(f, "clustergram"))

    assert out.count("data") == 56
    assert out.count("'x'") == 140
    assert out.count("'y'") == 140
    assert "cluster_labels" in out
    assert "count" in out
    assert "ratio" in out
    assert "size" in out


@pytest.mark.skipif(
    not RAPIDS,
    reason="RAPIDS not available.",
)
def test_bokeh_cuml():
    n_samples = 10
    n_features = 2

    n_clusters = 5
    random_state = 0

    device_data, device_labels = cuml.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    data = cudf.DataFrame(device_data)

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    f = clustergram.bokeh()
    out = str(json_item(f, "clustergram"))

    assert out.count("data") == 58
    assert out.count("'x'") == 145
    assert out.count("'y'") == 145
    assert "cluster_labels" in out
    assert "count" in out
    assert "ratio" in out
    assert "size" in out
