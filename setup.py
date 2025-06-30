from setuptools import setup, find_packages

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='graph2mat4abn',
    version='0.1',
    packages=find_packages(where='graph2mat4abn'),
    package_dir={'': 'graph2mat4abn'},
    install_requires=parse_requirements('requirements.txt'),
)