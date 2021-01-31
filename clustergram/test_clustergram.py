from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import pytest

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

    pd.testing.assert_series_equal(
        clustergram.labels.mean(),
        pd.Series([0.0, 0.2, 0.6, 1.8, 2.0, 2.2, 2.7], index=range(1, 8)),
    )
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
    assert expected == [np.mean(clustergram.cluster_centers[x]) for x in range(1, 8)]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == -0.21726383145643727
    assert clustergram.plot_data.mean().mean() == 2.3448529748438847


def test_sklearn_minibatchkmeans():
    clustergram = Clustergram(
        range(1, 8),
        backend="sklearn",
        method="minibatchkmeans",
        random_state=random_state,
    )
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.labels.mean(),
        pd.Series([0.0, 0.2, 0.6, 1.2, 2.0, 2.1, 2.6], index=range(1, 8)),
    )
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.35104516103195,
        2.7950777425566464,
        2.8277733533088534,
        2.522309579003138,
        2.3452078554650075,
        2.0602593594306526,
        2.2717280475246584,
    ]
    assert expected == [np.mean(clustergram.cluster_centers[x]) for x in range(1, 8)]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == -0.19729006113986378
    assert clustergram.plot_data.mean().mean() == 2.3478208185941405


def test_sklearn_gmm():
    clustergram = Clustergram(
        range(1, 8), backend="sklearn", method="gmm", random_state=random_state
    )
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.labels.mean(),
        pd.Series([0.0, 0.2, 0.6, 1.8, 2.0, 2.2, 2.7], index=range(1, 8)),
    )
    assert clustergram.labels.shape == (10, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        2.6841643400794335,
        2.4203136552626865,
        2.52976769800815,
        2.74946247582996,
        2.333257525098582,
        2.0444680314214723,
        2.2697082687156915,
    ]
    assert expected == [np.mean(clustergram.cluster_centers[x]) for x in range(1, 8)]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == -0.5578663671008622
    assert clustergram.plot_data.mean().mean() == 2.300054613897816


@pytest.mark.skipif(
    not RAPIDS, reason="RAPIDS not available.",
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
    clustergram = Clustergram(range(1, 10), backend="cuML", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.labels.mean().to_pandas(),
        pd.Series([0.0, 0.2, 0.6, 1.8, 2.0, 2.2, 2.7], index=range(1, 8)),
    )
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
    assert expected == [cp.mean(clustergram.cluster_centers[x]) for x in range(1, 8)]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == -0.21726383145643727
    assert clustergram.plot_data.mean().mean() == 2.3448529748438847

    # cupy array
    data = device_data

    clustergram = Clustergram(range(1, 10), backend="cuML")
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.labels.mean().to_pandas(),
        pd.Series([0.0, 0.2, 0.6, 1.8, 2.0, 2.2, 2.7], index=range(1, 8)),
    )
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
    assert expected == [cp.mean(clustergram.cluster_centers[x]) for x in range(1, 8)]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    ax.get_geometry() == (1, 1, 1)

    assert clustergram.plot_data_pca.mean().mean() == -0.21726383145643727
    assert clustergram.plot_data.mean().mean() == 2.3448529748438847


def test_errors():
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), backend="nonsense")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="nonsense")


def test_repr_():
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
