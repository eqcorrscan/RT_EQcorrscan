from __future__ import print_function
try:
    # use setuptools if we can
    from setuptools import setup
    using_setuptools = True
except ImportError:
    from distutils.core import setup
    using_setuptools = False

import glob
import os
import sys
import shutil

with open("rt_eqcorrscan/__init__.py", "r") as init_file:
    version_line = [line for line in init_file
                    if '__version__' in line][0]
VERSION = version_line.split()[-1].split("'")[1]

long_description = '''
Real-time EQcorrscan: Real-time wrappers for EQcorrscan's earthquake detection
methods.
'''

scriptfiles = [f for f in glob.glob("scripts/*") if os.path.isfile(f)]


def setup_package():

    # Figure out whether to add ``*_requires = ['numpy']``.
    build_requires = []
    try:
        import numpy
    except ImportError:
        build_requires = ['numpy>=1.6, <2.0']

    install_requires = [
        'numpy>=1.12', 'matplotlib>=1.3.0', 'scipy>=0.18', 'LatLon',
        'bottleneck', 'bokeh', 'obspy>=1.0.3', 'h5py', 'eqcorrscan>=0.3.0',
        'obsplus']

    setup_args = {
        'name': 'RT-EQcorrscan',
        'version': VERSION,
        'description': 'RT-EQcorrscan - Real-time matched-filter detection',
        'long_description': long_description,
        'url': 'https://github.com/calum-chamberlain/EQcorrscan',
        'author': 'Calum Chamberlain',
        'author_email': 'calum.chamberlain@vuw.ac.nz',
        'license': 'LGPL',
        'classifiers': [
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: GNU Library or Lesser General Public '
            'License (LGPL)',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        'keywords': 'real-time earthquake correlation detection match-filter',
        'scripts': scriptfiles,
        'install_requires': install_requires,
        'setup_requires': ['pytest-runner'],
        'tests_require': ['pytest>=2.0.0', 'pytest-cov', 'pytest-pep8',
                          'pytest-xdist', 'pytest-rerunfailures',
                          'obspy>=1.1.0'],
    }

    if using_setuptools:
        setup_args['setup_requires'] = build_requires
        setup_args['install_requires'] = install_requires

    if len(sys.argv) >= 2 and (
        '--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean')):
        # For these actions, NumPy is not required.
        pass
    else:
        setup_args['packages'] = [
            'rt_eqcorrscan', 'rt_eqcorrscan.config', 'rt_eqcorrscan.database',
            'rt_eqcorrscan.event_trigger', 'rt_eqcorrscan.plotting',
            'rt_eqcorrscan.reactor', 'rt_eqcorrscan.streaming',
            'rt_eqcorrscan.streaming.clients']
    if os.path.isdir("build"):
        shutil.rmtree("build")
    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
