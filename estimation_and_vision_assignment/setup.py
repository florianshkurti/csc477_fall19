#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    # #  don't do this unless you want a globally visible script
    # scripts=['bin/myscript'], 
    packages=['estimation_and_vision'],
    package_dir={'': 'python'},
    scripts=['python/stereo_disparity_map.py',
             'python/monte_carlo_localization.py',]
)

setup(**d)
