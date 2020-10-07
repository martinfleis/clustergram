# Clustergram

## Visualization and diagnostics for cluster analysis

Clustergram is a diagram proposed by Matthias Schonlau in his paper *[The clustergram: A graph for visualizing hierarchical and nonhierarchical cluster analyses](https://journals.sagepub.com/doi/10.1177/1536867X0200200405)*.

> In hierarchical cluster analysis, dendrograms are used to visualize how clusters are formed. I propose an alternative graph called a “clustergram” to examine how cluster members are assigned to clusters as the number of clusters increases. This graph is useful in exploratory analysis for nonhierarchical clustering algorithms such as k-means and for hierarchical cluster algorithms when the number of observations is large enough to make dendrograms impractical.

The clustergram was later implemented in R by [Tal Galili](https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/), who also gives a thorough explanation of the concept.

This is a Python translation of Tal's script written for `scikit-learn` and RAPIDS `cuML` implementations of K-Means clustering (as of v0.1).

## Getting started

```shell
pip install git+git://github.com/martinfleis/clustergram.git
```

The example of clustergram on Palmer penguins dataset:

```python
import seaborn
df = seaborn.load_dataset('penguins')
```

First we have to select numerical data and scale them.

```python
from sklearn.preprocessing import scale
data = scale(df.drop(columns=['species', 'island', 'sex']).dropna())
```

And then we can simply pass the data to `clustergram`.
```python
from clustergram import clustergram
clustergram(data, range(1, 8))
```

![Default clustergram](doc/default.png)

## Styling

`clustergram` returns matplotlib axis and can be fully customised as any other matplotlib plot.

```python
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

clustergram(
    data,
    range(1, 8),
    ax=ax,
    size=0.5,
    linewidth=0.05,
    cluster_style={"color": "lightblue", "edgecolor": "black"},
    line_style={"color": "red", "linestyle": "-."},
)
```
![Colored clustergram](doc/colors.png)

## Mean options

On the `y` axis, a clustergram can use mean values as in the original paper by Matthias Schonlau or PCA weighted mean values as in the implementation by Tal Galili.

```python
fig, ax = plt.subplots(figsize=(12, 8))
clustergram(data, range(1, 8), ax=ax, pca_weighted=True)
```
![Default clustergram](doc/pca_true.png)

```python
fig, ax = plt.subplots(figsize=(12, 8))
clustergram(data, range(1, 8), ax=ax, pca_weighted=False)
```
![Default clustergram](doc/pca_false.png)


## Scikit-learn and RAPIDS cuML backends

Clustergram offers two backends for the computation - `scikit-learn` which uses CPU and RAPIDS.AI `cuML`, which uses GPU. Note that both are optional dependencies, but you will need at least one of them to generate clustergram.

Using scikit-learn (default):

```python
clustergram(data, range(1, 8), backend='sklearn')
```

Using cuML (default):

```python
clustergram(data, range(1, 8), backend='cuML')
```

`data` can be all data types supported by the selected backend (including `cudf.DataFrame` with `cuML` backend).

## References
Schonlau M. The clustergram: a graph for visualizing hierarchical and non-hierarchical cluster analyses. The Stata Journal, 2002; 2 (4):391-402.

Schonlau M. Visualizing Hierarchical and Non-Hierarchical Cluster Analyses with Clustergrams. Computational Statistics: 2004; 19(1):95-111.

https://www.r-statistics.com/2010/06/clustergram-visualization-and-diagnostics-for-cluster-analysis-r-code/