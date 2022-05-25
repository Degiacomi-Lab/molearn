# python setup.py install
#
# to build documentation:
# sphinx-build -b html doc doc/html

import os
import shutil
from setuptools import setup, find_packages


packages=find_packages(where='src')
print('packages being: ', packages)
setup(
    name = 'molearn',
    version='1.0.0',
    package_data={"":["*.dat"]},
    packages=packages,
    package_dir={"":"src"},
)
