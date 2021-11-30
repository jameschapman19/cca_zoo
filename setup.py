from setuptools import setup,  find_packages

with open("README.md") as fp:
    long_description = fp.read()

def setup_package():
    data = dict(
    name='ccagame',
    version='0',
    packages=find_packages(exclude=['test']),
    url='https://github.com/jameschapman19/ccagame',
    license='',
    author='James Chapman',
    author_email='james.chapman.19@ucl.ac.uk',
    description='PLS/CCA/PCA formulated as games',
    long_description=long_description,
    install_requires=[
        "pandas~=1.3.4",
        "wandb~=0.12.5",
        "numpy~=1.21.3",
        "jax~=0.2.24",
        "jaxlib~=0.1.73",
        "scikit-learn~=1.0",
        "scipy~=1.7.1",    
        "optax~=0.1.0",
        "tensorflow~=2.4.1",
    ]
    )

    setup(**data)

if __name__ == "__main__":
    setup_package()

