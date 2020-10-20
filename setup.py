from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cca_zoo',
    version='1.0.1',
    packages=find_packages(),
    url='https://github.com/jameschapman19/cca_zoo',
    license='MIT',
    author='jameschapman',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    description='',
    install_requires=['matplotlib>=3.1.1',
                        'numpy>=1.17.4',
                        'pandas>=1.0.3',
                        'scikit-learn>=0.22.1',
                        'scipy>=1.4.1',
                        'seaborn>=0.10.1',
                        'torch>=1.5.0',
                        'tqdm>=4.47.0'])
