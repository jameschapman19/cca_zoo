from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("./requirements/basic.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

EXTRA_PACKAGES = {}
with open("./requirements/deep.txt", "r") as f:
    EXTRA_PACKAGES["deep"] = f.read()
with open("./requirements/probabilistic.txt", "r") as f:
    EXTRA_PACKAGES["probabilistic"] = f.read()
EXTRA_PACKAGES["all"] = EXTRA_PACKAGES["deep"] + "\n" + EXTRA_PACKAGES["probabilistic"]

setup(
    name="cca_zoo",
    version="0.0.0",
    include_package_data=True,
    keywords="cca",
    packages=find_packages(),
    url="https://github.com/jameschapman19/cca_zoo",
    license="MIT",
    author="jameschapman",
    description=(
        "Canonical Correlation Analysis Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic methods in a scikit-learn style framework"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="james.chapman.19@ucl.ac.uk",
    python_requires=">=3.6",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    test_suite="test",
    tests_require=[],
)
