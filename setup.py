from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['numpy',
                'torch',
                'scikit-learn',
                'scipy',
                'matplotlib',
                'seaborn',
                'Pillow',
                'torchvision',
                'pandas',
                'mvlearn']

setup(
    name='cca_zoo',
    version='1.1.15',
    packages=find_packages(),
    url='https://github.com/jameschapman19/cca_zoo',
    license='MIT',
    author='jameschapman',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    description='',
    install_requires=requirements,
    test_suite='tests',
    tests_require=[],
)
