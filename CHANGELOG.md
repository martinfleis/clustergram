# Changelog

## Version 0.7.0 (January 15, 2023)

__Enhancements__:

- ENH: allow weighting by a custom principal component (#35)

__Compatibility notes__:

- ``clustergram`` now requires Python 3.8
- RAPIDS.AI implementation has been tested with version 22.12.

__Minor notes__:

- ``examples`` dictionary has been removed. Refer to the notebooks in the documentation.

## Version 0.6.0 (November 21, 2021)

__Enhancements__:

- ENH: optionally measure BIC during GMM (#21)

__Bug fixes__:

- BUG: ``cuML`` non-weighted plot fix (#25)

## Version 0.5.1 (May 24, 2021)

Fix for ``from_data`` method with non-default indices.

__Bugs__:

- BUG: cluster centers empty due to index mismatch (#19)

## Version 0.5.0 (May 11, 2021)

``clustergram`` now supports interactive plotting using a new `.bokeh()` method based on ``BokehJS``. It
can be handy for exploration of larger and more complex clustergrams or those with significant outliers.

__Enhancements__:

- ENH: support interactive ``bokeh`` plots (#14)
- ENH: skip ``k=1`` in K-Means implementations (#18)

- documentation restructuring

## Version 0.4.0 (April 27, 2021)

Spring comes with native hierarchical clustering and the ability to create clustergam from a manual input.

__Enhancements__:

- ENH: support hierarchical clustering using ``scipy`` (#11)
- ENH: ``from_data`` and ``from_centers`` methods (#12)

## Version 0.3.0 (January 31, 2021)

API changes:

- ``pca_weighted`` is now keyword of ``Clustergram.plot()`` not ``__init__``.

Enhancements:

- Support ``MiniBatchKMeans`` (``scikit-learn``)
- Custom ``__repr__``
- Expose cluster labels obtained during the loop
- Expose cluster centers
- Silhouette score
- Calinski and Harabasz score
- Davies-Bouldin score

## Version 0.2.0 (December 21, 2020)

Version 0.2.0 brings support of Gaussian Mixture Models (using ``scikit-learn``) and few minor changes.

Enhancements:

- Gaussian Mixture Model support (#4)
- Verbosity - Clustergram now indicates the progress
- Additional arguments can be passed to the PCA object

Bug fixes:

- BUG: avoid LinAlgError: singular matrix
