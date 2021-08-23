from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("./requirements/basic.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

EXTRA_PACKAGES = {}
with open("./requirements/deep.txt", "r") as f:
    EXTRA_PACKAGES['deep'] = f.read()
with open("./requirements/probabilistic.txt", "r") as f:
    EXTRA_PACKAGES['probabilistic'] = f.read()

setup(
    name='cca_zoo',
    version='1.7.13',
    include_package_data=True,
    keywords='cca',
    packages=find_packages(),
    url='https://github.com/jameschapman19/cca_zoo',
    license='MIT',
    author='jameschapman',
    description=(
        'Canonical Correlation Analysis Zoo: CCA, GCCA, MCCA, DCCA, DGCCA, DVCCA, DCCAE, KCCA and regularised variants including sparse CCA , ridge CCA and elastic CCA'
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    python_requires='>=3.6',
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    test_suite='tests',
    tests_require=[],
)
