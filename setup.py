from setuptools import setup

setup(
    name='pyCCA',
    version='1',
    packages=['PyCCA'],
    url='https://github.com/jameschapman19/CCA_methods',
    license='MIT',
    author='jameschapman',
    author_email='james.chapman.19@ucl.ac.uk',
    description='',
    install_requires=['matplotlib==3.1.1',
                        'numpy==1.17.4',
                        'pandas==1.0.3',
                        'scikit-learn==0.22.1',
                        'scipy==1.4.1',
                        'seaborn==0.10.1',
                        'torch==1.5.0',
                        'torchvision==0.6.0a0+82fd1c8',
                        'tqdm==4.47.0'])
