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
import rt_eqcorrscan

VERSION = rt_eqcorrscan.__version__

long_description = '''
Real-time EQcorrscan: Real-time wrappers for EQcorrscan's earthquake detection
methods.
'''

scriptfiles = glob.glob("rt_eqcorrscan/scripts/*.py")


def setup_package():

    # Figure out whether to add ``*_requires = ['numpy']``.
    build_requires = []
    try:
        import numpy
    except ImportError:
        build_requires = ['numpy>=1.6, <2.0']

    install_requires = ['matplotlib>=1.3.0', 'scipy>=0.18', 'LatLon',
                        'bottleneck', 'obspy>=1.0.3', 'numpy>=1.12',
                        'h5py', "eqcorrscan>=0.3.0", 'bokeh']
    install_requires.extend(build_requires)

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
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
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
            'rt_eqcorrscan', 'rt_eqcorrscan.utils', 'rt_eqcorrscan.core',
            'rt_eqcorrscan.plotting']
    if os.path.isdir("build"):
        shutil.rmtree("build")
    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
