Changelog
=========

Version 0.4.0 (April 27, 2021)
------------------------------

Spring comes with native hierarchical clustering and the ability to create clustergam from a manual input.

Enhancements:

- ENH: support hierarchical clustering using scipy (#11)
- ENH: from_data and from_centers methods (#12)


Version 0.3.0 (January 31, 2021)
--------------------------------

API chages:

- ``pca_weighted`` is now keyword of ``Clustergram.plot()`` not init.

Enhancements:

- Support ``MiniBatchKMeans`` (sklearn)
- Custom __repr__
- Expose cluster labels obtained during the loop
- Expose cluster centers
- Silhouette score
- Calinski and Harabasz score
- Davies-Bouldin score


Version 0.2.0 (December 21, 2020)
---------------------------------

Version 0.2.0 brings support of Gaussian Mixture Models (using scikit-learn) and few minor changes.

Enhancements:

- Gaussian Mixture Model support (#4)
- Verbosity - Clustergram now indicates the progress
- Additional arguments can be passed to the PCA object

Bug fixes:

- BUG: avoid LinAlgError: singular matrix
