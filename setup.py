import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustergram",
    version="0.2.1",
    author="Martin Fleischmann",
    author_email="martin@martinfleischmann.net",
    description="Clustergram - visualization and diagnostics for cluster analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinfleis/clustergram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
    install_requires=["matplotlib", "numpy", "pandas"],
)
