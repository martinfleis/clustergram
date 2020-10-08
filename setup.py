import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clustergram",
    version="0.1.0",
    author="Martin Fleischmann",
    author_email="martin@martinfleischmann.net",
    description="Clustergram - visualization and diagnostics for cluster analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinfleis/clustergram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["matplotlib"],
)
