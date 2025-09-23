from setuptools import setup, find_packages

setup(
    name='graph2mat4abn',
    version='0.1',
    package_dir={'': 'src'},                             
    packages=find_packages(where='src'),
    install_requires=[l.strip() for l in open('requirements.txt')],
)
