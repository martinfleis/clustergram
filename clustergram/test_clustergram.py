from sklearn.datasets import make_blobs
import pandas as pd
import pytest

try:
    import cudf
    import cuml

    RAPIDS = True
except (ImportError, ModuleNotFoundError):
    RAPIDS = False

from clustergram import Clustergram


def test_sklearn():
    n_samples = 1000
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

    clustergram = Clustergram(range(1, 10), backend="sklearn")
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.means.mean(),
        pd.Series(
            [
                -0.30042205,
                -0.30042205,
                -0.30042205,
                -0.30042205,
                -0.30042205,
                -0.30046431,
                -0.3003699,
                -0.30086877,
                -0.30016977,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    ax = clustergram.plot()
    ax.get_geometry() == (1, 1, 1)

    clustergram = Clustergram(range(1, 10), backend="sklearn", pca_weighted=False)
    clustergram.fit(device_data)

    pd.testing.assert_series_equal(
        clustergram.means.mean(),
        pd.Series(
            [
                2.05533664,
                2.05533664,
                2.05533664,
                2.05533664,
                2.05533664,
                2.05522513,
                2.05582383,
                2.05585776,
                2.05591782,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    ax = clustergram.plot()
    ax.get_geometry() == (1, 1, 1)


@pytest.mark.skipif(
    not RAPIDS, reason="RAPIDS not available.",
)
def test_cuml():
    n_samples = 1000
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

    clustergram = Clustergram(range(1, 10), backend="cuML")
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.means.mean().to_pandas(),
        pd.Series(
            [
                0.30042205,
                0.30042205,
                0.30042205,
                0.30042205,
                0.30042205,
                0.30046431,
                0.3003699,
                0.30086877,
                0.30016977,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    data = device_data

    clustergram = Clustergram(range(1, 10), backend="cuML")
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.means.mean().to_pandas(),
        pd.Series(
            [
                0.30042205,
                0.30042205,
                0.30042205,
                0.30042205,
                0.30042205,
                0.30046431,
                0.3003699,
                0.30086877,
                0.30016977,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    device_data, device_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        cluster_std=0.1,
    )

    clustergram = Clustergram(range(1, 10), backend="cuML", pca_weighted=False)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.means.mean().to_pandas(),
        pd.Series(
            [
                2.31314663,
                2.31314663,
                2.31314663,
                2.31314663,
                2.31314663,
                2.31353356,
                2.31314663,
                2.31326447,
                2.31284071,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    clustergram = Clustergram(range(1, 10), backend="cuML", pca_weighted=False)
    clustergram.fit(cudf.DataFrame(device_data))

    pd.testing.assert_series_equal(
        clustergram.means.mean().to_pandas(),
        pd.Series(
            [
                2.31314663,
                2.31314663,
                2.31314663,
                2.31314663,
                2.31314663,
                2.31353356,
                2.31314663,
                2.31326447,
                2.31284071,
            ],
            index=list(range(1, 10)),
        ),
        rtol=6,
    )

    ax = clustergram.plot()
    ax.get_geometry() == (1, 1, 1)


def test_errors():
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), backend="nonsense")
    with pytest.raises(ValueError):
        Clustergram(range(1, 3), method="nonsense")

