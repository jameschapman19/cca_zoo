from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cca_zoo',
    version='1.1.1',
    packages=find_packages(),
    url='https://github.com/jameschapman19/cca_zoo',
    license='MIT',
    author='jameschapman',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    description='',
    install_requires=['torch>=1.7.0',
                        'numpy>=1.17.4',
                        'scikit-learn>=0.22.1',
                        'scipy>=1.4.1',
                        'hyperopt>=0.2.5',
                        'matplotlib>=3.3.2',
                        'seaborn>=0.10.1',
                        'torchvision>=0.8.1',
                        'pandas>=1.0.3',
                        'mvlearn>=0.3.0'])
