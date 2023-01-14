from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import pytest
from bokeh.embed import json_item
from pandas.testing import assert_series_equal


try:
    import cudf
    import cuml
    import cupy as cp

    RAPIDS = True
except (ImportError, ModuleNotFoundError):
    RAPIDS = False

from clustergram import Clustergram

n_samples = 100
n_features = 2

n_clusters = 8
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
    assert clustergram.labels.shape == (100, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        1.439891622331535,
        -2.809248339265837,
        -0.9554163965815223,
        0.15829646201444203,
        0.626698921291375,
        0.9155105021035385,
        1.0238657347680074,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 46

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 46

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -2.095277953205114, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        1.4398916223315354, rel=1e-15
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
    assert clustergram.labels.shape == (100, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        1.439891622331535,
        1.5942431676314943,
        -0.9391362578715787,
        0.16457587659721762,
        0.7988407523191436,
        0.9230637622852088,
        1.0250449911587773,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 45

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 45

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -2.153978086091386, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        1.477158426841248, rel=1e-15
    )


def test_sklearn_gmm():
    clustergram = Clustergram(
        range(1, 8), backend="sklearn", method="gmm", random_state=random_state
    )
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (100, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        1.4886908509157464,
        -2.8599808770366817,
        -0.8823883211732156,
        0.18416419702253917,
        0.08229356227237798,
        0.6537149985640699,
        0.927345926721354,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-6)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -1.9629843968429452, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        1.3321040444661392, rel=1e-15
    )


def test_bic():
    clustergram = Clustergram(
        range(1, 8),
        backend="sklearn",
        method="gmm",
        random_state=random_state,
        bic=True,
    )
    clustergram.fit(data)

    expected = pd.Series(
        [
            1226.7924019554766,
            948.6374834781362,
            800.1788609508928,
            687.5987056807201,
            497.2770114251739,
            402.1340827435864,
            306.6669136240255,
        ],
        index=range(1, 8),
    )

    assert_series_equal(expected, clustergram.bic, rtol=1e-6)

    clustergram = Clustergram(
        range(1, 8),
        backend="sklearn",
        method="gmm",
        random_state=random_state,
        bic=False,
    )
    clustergram.fit(data)
    assert hasattr(clustergram, "bic") is False


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
        0.9148379012942314,
        0.6835291385650635,
        0.9405179619789124,
        0.8763175010681152,
        1.5546628013253212,
        1.2617384965221086,
        0.7542384501014437,
    ]
    assert expected == [
        pytest.approx(float(clustergram.cluster_centers[x].mean().mean()), rel=1e-3)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        1.3444129803913, rel=1e-3
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        0.9148379244974681, rel=1e-3
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
        0.9148379012942314,
        0.6835291385650635,
        0.9405179619789124,
        0.8763175010681152,
        1.5546628013253212,
        1.2617384965221086,
        0.7542384501014437,
    ]
    assert expected == [
        pytest.approx(float(cp.mean(clustergram.cluster_centers[x])), rel=1e-6)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        1.344412697695078, rel=1e-3
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        0.9148379244974681, rel=1e-3
    )


def test_hierarchical():
    clustergram = Clustergram(range(1, 8), method="hierarchical")
    clustergram.fit(data)

    for i in range(1, 8):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (100, 7)
    assert clustergram.labels.notna().all().all()

    expected = [
        1.4398916223315354,
        -2.8092483392658374,
        -0.7499055624802712,
        0.28659658912247143,
        0.7961494117071617,
        0.9155105021035381,
        1.023865734768007,
    ]
    assert expected == [
        pytest.approx(np.mean(clustergram.cluster_centers[x]), rel=1e-12)
        for x in range(1, 8)
    ]

    assert clustergram.plot_data_pca.empty
    ax = clustergram.plot(pca_kwargs=dict(random_state=random_state))
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 44

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -2.0952779532051142, rel=1e-15
    )
    assert clustergram.plot_data.mean().mean() == pytest.approx(
        1.4398916223315354, rel=1e-15
    )


def test_hierarchical_array():
    clustergram = Clustergram(method="hierarchical", k_range=range(1, 10))
    clustergram.fit(data.values)

    for i in range(1, 10):
        assert clustergram.labels[i].nunique() == i
    assert clustergram.labels.shape == (100, 9)
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
            [
                0.6754810302465651,
                0.6277858262368159,
                0.6728079183937916,
                0.7092450515302072,
                0.8001963572359172,
                0.8798871538184535,
            ],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.silhouette,
        pd.Series(
            [
                0.6754810302465651,
                0.6277858262368159,
                0.6728079183937916,
                0.7092450515302072,
                0.8001963572359172,
                0.8798871538184535,
            ],
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
            [0.63275963, 0.5933514, 0.7809184, 0.8807362, 0.68701756, 0.4919311],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
        check_dtype=False,
    )

    clustergram = Clustergram(range(1, 8), backend="cuML", random_state=random_state)
    clustergram.fit(device_data)

    pd.testing.assert_series_equal(
        clustergram.silhouette_score(),
        pd.Series(
            [0.63275963, 0.5933514, 0.7809184, 0.8807362, 0.68701756, 0.4919311],
            index=list(range(2, 8)),
            name="silhouette_score",
        ),
        check_dtype=False,
    )


def test_calinski_harabasz_score():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz_score(),
        pd.Series(
            [
                114.18545531981596,
                259.8218744719872,
                446.25054149041324,
                586.3857013614834,
                916.5220549808022,
                1689.4091019412879,
            ],
            index=list(range(2, 8)),
            name="calinski_harabasz_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.calinski_harabasz,
        pd.Series(
            [
                114.18545531981596,
                259.8218744719872,
                446.25054149041324,
                586.3857013614834,
                916.5220549808022,
                1689.4091019412879,
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
                17.854664875826852,
                18.993060869559063,
                25.53897801880369,
                10495.855575243557,
                10895.935616041483,
                10449.035861758717,
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
                17.854664875826852,
                18.993060869559063,
                25.53897801880369,
                10495.855575243557,
                10895.935616041483,
                10449.035861758717,
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
            [
                0.2945752391269888,
                0.5101512437048275,
                0.4762688744525792,
                0.4822529450245402,
                0.3533377436714937,
                0.21391254262995393,
            ],
            index=list(range(2, 8)),
            name="davies_bouldin_score",
        ),
    )

    pd.testing.assert_series_equal(
        clustergram.davies_bouldin,
        pd.Series(
            [
                0.2945752391269888,
                0.5101512437048275,
                0.4762688744525792,
                0.4822529450245402,
                0.3533377436714937,
                0.21391254262995393,
            ],
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
                0.3573296075971386,
                0.7673811855139047,
                0.4520342597085474,
                0.02258593626130912,
                0.01451002792630246,
                0.00967011650130667,
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
                0.3573296075971386,
                0.7673811855139047,
                0.4520342597085474,
                0.02258593626130912,
                0.01451002792630246,
                0.00967011650130667,
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
    assert len(ax.get_children()) == 18

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 18

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
    assert len(ax.get_children()) == 18

    assert clustergram.plot_data.empty
    ax = clustergram.plot(pca_weighted=False)
    assert len(ax.get_children()) == 18

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
    assert len(ax.get_children()) == 18

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
    assert len(ax.get_children()) == 18

    assert clustergram.plot_data_pca.mean().mean() == pytest.approx(
        -0.15713484026367722, rel=1e-15
    )


def test_bokeh():
    clustergram = Clustergram(range(1, 8), backend="sklearn", random_state=random_state)
    clustergram.fit(data)

    f = clustergram.bokeh(pca_kwargs=dict(random_state=random_state))
    out = str(json_item(f, "clustergram"))

    assert out.count("data") == 60
    assert "cluster_labels" in out
    assert "count" in out
    assert "ratio" in out
    assert "size" in out

    f = clustergram.bokeh(pca_weighted=False)
    out = str(json_item(f, "clustergram"))

    assert out.count("data") == 60
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

    assert out.count("data") == 56
    assert "cluster_labels" in out
    assert "count" in out
    assert "ratio" in out
    assert "size" in out
