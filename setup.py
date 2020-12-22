from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements=['numpy~=1.19.3',
 'torch~=1.7.1',
'scikit-learn~=0.23.2',
 'scipy~=1.5.4',
 'matplotlib~=3.3.3',
 'seaborn~=0.11.0',
 'Pillow~=8.0.1',
 'torchvision~=0.8.2',
 'pandas~=1.1.5',
 'mvlearn~=0.4.1']

setup(
    name='cca_zoo',
    version='1.1.6',
    packages=find_packages(),
    url='https://github.com/jameschapman19/cca_zoo',
    license='MIT',
    author='jameschapman',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    description='',
    install_requires=requirements)
