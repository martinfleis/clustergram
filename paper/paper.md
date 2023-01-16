---
title: 'Clustergram: Visualization and diagnostics for cluster analysis'
tags:
  - Python
  - clustering
  - unsupervised classification
  - data exploration
authors:
  - name: Martin Fleischmann
    orcid: 0000-0003-3319-3366
    affiliation: 1
affiliations:
 - name: Department of Social Geography and Regional Development, Charles University
   index: 1
date: 16 January 2023
bibliography: paper.bib
---

# Summary

Given a heterogeneous group of observations, research often tries to find more
homogenous groups within them. Typical is the use of clustering algorithms determining
these groups based on statistical similarity. While there is an extensive range of
algorithms to be chosen from, they often share one specific limitation - the algorithm
itself will not determine the optimal number of clusters a group of observations shall
be divided into. This paper presents a Python package named `clustergram` providing
tools to analyze the clustering solutions and visualize the behavior of observations in
relation to a tested range of options for the number of classes, enabling a deeper
understanding of the behavior of observations splitting into classes and better-informed
decisions on the optimal number of classes.

The situation the package is dealing with can be illustrated on one of the most commonly
used clustering algorithms, K-Means. The algorithm first sets a pre-defined number of
random seeds and attempts to split the data into the same number of classes, searching
for the optimal seed locations providing the best split between the groups. However, the
number of seed locations needs to be defined by a researcher and is usually unknown. The
clustering solution is therefore created for a range of viable solutions (usually from 2
to N) that are compared and assessed based on various criteria, be it a so-called
_"elbow plot"_ of silhouette score looking for the "elbow" on a curve or a related
silhouette analysis, or using other evaluation metrics. Most of them have in common that
they treat each clustering option separately, without a relation between, e.g., when
testing 3 and 4 clusters, the behavior of observations between these two options is not
considered. To alleviate the situation and shed more light on the dynamics of
_reshuffling_ of observations between clusters, @schonlau2002clustergram proposed a new
visual method called _"clustergram"_.

Clustergrams take the shape of a hierarchical diagram displaying a range of clustering
options (number of clusters) on (usually) the X-axis and cluster centers for each
solution on the Y-axis. Furthermore, there is an indication of a number of observations
shifting between clusters, so we can see how large a portion of cluster A from a
2-cluster solution goes to cluster B of a 3-cluster solution, for example. This
visualization uncovers the hierarchical nature of range-based series of clustering
solutions and enables researchers to determine the optimal number of classes based on
the illustrated behavior of observations as shown in figures \autoref{fig:mean}, and
\autoref{fig:pca} further explained below.

The Python package presented in this paper provides tools to create and explore
clustergrams in Python based on a number of built-in clustering algorithms but also on
external input resulting from other algorithms. The API is organized around a single
overarching `clustergram.Clustergram` class designed around scikit-learn's API style
[@scikit-learn] with initialization of the class with the specification of arguments and
the `fit` method, making it familiar to existing users of scikit-learn and
similarly-designed packages. In its core, the class expects a selection of a range of
solutions to be tested (`k_range`) from 1 to N, a selection of clustering algorithm
(`method`) and a specification of a backend used for computation. Here, `clustergram`
offers a choice between backends written to run on a CPU (`scikit-learn` for K-Means,
Mini-batch K-Means and Gaussian Mixture Models, `scipy` [@2020SciPy-NMeth] for
hierarchical (or agglomerative) algorithms) or a GPU (`cuML` [@raschka2020machine] for
K-Means), where the GPU path is computing both clustering and the underlying data for
clustergram visualization on GPU, minimizing the need of data transfer between both.
Furthermore, suppose none of the built-in options is suited for a set use case. In that
case, the clustergram data structure can be created either from original data and labels
for individual cluster solutions (`from_data()` method) or from cluster centers
(`from_centers()` method), depending on the information obtainable from the selected
external clustering algorithm.

![Clustergram based on the K-Means clustering algorithm as implemented in the
scikit-learn package based on Palmer penguins dataset (@palmerpenguins). The cluster
centroids are showing the non-weighted mean values as proposed in the original paper by
@schonlau2002clustergram.\label{fig:mean}](mean.svg)

Once the series of cluster solutions is generated, it is time to compute and generate
clustergram diagrams for plotting functionality. The package offers two different ways
of computing clustergram values. The first case shown in figure \autoref{fig:mean}
follows the original proposal by @schonlau2002clustergram and uses the means of cluster
centroids (i.e. a mean of means of features) to plot on Y-axis. However, as later noted
by @2010Clustergram, that does not necessarily provide the best overview of the
behavior. Therefore, there is another (default) option weighted the means of cluster
centroids by the first principal component derived from the complete dataset, shown in
figure \autoref{fig:pca} based on the same set of clustering solutions. Moreover,
weighting by any other principal component is also available if a researcher needs
further exploration. Due to the potential high computation cost of principal components
and weighted cluster centroids, all the values are cached once computed, meaning that
only the first plotting call triggers the computation.

![Clustergram based on the K-Means clustering algorithm as implemented in the
scikit-learn package based on Palmer penguins dataset (@palmerpenguins), together with
the additional metrics of cluster fit generated by the package. The cluster centroids
are weighted by the first principal component to enhance the distinction between the
branches of the dendrogram.\label{fig:pca}](pca.svg)

The plotting is implemented in two different options, showing the same diagram but one
as a static `matplotlib` [@hunter2007Matplotlib] figure while the other as an
interactive JavaScript-based visualization based on `bokeh` [@bokehteamBokeh]. The
latter is especially beneficial as it offers direct links of cluster centroids within
the diagram to individual labels allowing very granular back-and-forth diagnostics of
the clustering behavior.

Since the selection of an optimal number of classes is a non-trivial exercise and shall
not, in the ideal case, be left to a single method or metric, `clustergram` natively
allows computation of additional metrics of cluster fit (Silhouette score,
Calinski-Harabasz score, Davies-Bouldin score) directly from the main class using the
implementation available in the `scikit-learn`.

# Statement of need

As the problem `clustergram` helps resolve is not closed, there is a need for additional
methods beyond the elbow plot and other traditionally used ways. It is clearly indicated
by the constant citation level of the original set of papers by Schonlau
[@schonlau2002clustergram; @schonlau2004Visualizing]. Arguably, this has been limited by
the lack of ready-to-use implementation of the technique in the modern data science
pipelines as the @schonlau2002clustergram's code has been written in 2002 for STATA and
the only other version has been explored in a blog post by @2010Clustergram
experimenting with the minimal (as well as unpackaged and unmaintained) R
implementation. Since the first release of `clustergram` in November 2020 (and
publishing a blog post in April 2021), the package has been used in at least seven
academic publications, ranging from the classification of geographical areas based on
form and function [@arribas-bel2022Spatiala; @fleischmann2022Geographicala;
@samardzhiev2022Functionala], clustering of the latent representation from convolutional
neural networks [@singleton2022Estimatinga], classification of high Arctic lakes
[@urbanski2022Monitoring] to facility reliability assessment [@stewart2022Addressing]
and genomic data science [@ma2022Abstract]. Since none of these directly cite the
software, it is likely an incomplete overview. While researchers can still use the
traditional set of metrics to estimate the optimal number of classes, none, including
clustergram, is the ultimate answer without any drawbacks. What makes clustergram unique
is the reflection of the dynamics of the sequence of solutions and the visualization of
the behavior of observations within it.

# Acknowledgements

The author kindly acknowledges the funding of the initial development by the UK’s
Economic and Social Research Council through the project “Learning an urban grammar from
satellite data through AI”, project reference ES/ T005238/1. Further appreciation
belongs to Tal Galili, who popularized the method in the R world in 2010 and from whom
clustergram borrowed its subtitle, _Visualization and diagnostics for cluster analysis_.

# References
