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

# Scripts need to have a `main` function
scriptfiles = [
    f for f in glob.glob(os.path.join("rt_eqcorrscan", "console_scripts", "*"))
    if os.path.isfile(f) and os.path.split(f)[-1] != "__init__.py"]
scriptnames = [
    os.path.splitext(os.path.split(f)[-1])[0].replace("_", "-")
    for f in scriptfiles]
console_entry_points = [
    f"rteqcorrscan-{name}=rt_eqcorrscan.console_scripts."
    f"{name.replace('-', '_')}:main" for name in scriptnames]

# Plugin entry points
plugin_scriptfiles = [
    f for f in glob.glob(os.path.join(
        "rt_eqcorrscan", "plugins", "console_scripts", "*"))
    if os.path.isfile(f) and os.path.split(f)[-1] != "__init__.py"]
plugin_scriptnames = [
    os.path.splitext(os.path.split(f)[-1])[0].replace("_", "-")
    for f in plugin_scriptfiles]
console_entry_points.extend([
    f"rteqcorrscan-plugin-{name}=rt_eqcorrscan.plugins.console_scripts."
    f"{name.replace('-', '_')}:main"
    for name in plugin_scriptnames])


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
        'obsplus', 'tqdm']

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
            'Programming Language :: Python :: 3.8',
        ],
        'keywords': 'real-time earthquake correlation detection match-filter',
        'entry_points': {'console_scripts': console_entry_points},
        'install_requires': install_requires,
        'setup_requires': ['pytest-runner'],
        'tests_require': ['pytest>=2.0.0', 'pytest-cov', 'pytest-pep8',
                          'pytest-xdist', 'pytest-rerunfailures',
                          'obspy>=1.1.0'],
        # 'package_data': {
        #     'rt_eqcorrscan/plugins/relocation':
        #         ['rt_eqcorrscan/plugins/relocation/run_growclust3D.jl',
        #          'rt_eqcorrscan/plugins/relocation/vmodel.txt',
        #          'rt_eqcorrscan/plugins/relocation/growclust.inp',
        #          'rt_eqcorrscan/plugins/relocation/README.md']},
        'include_package_data': True
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
            'rt_eqcorrscan.streaming.clients', 'rt_eqcorrscan.console_scripts',
            'rt_eqcorrscan.plugins', 'rt_eqcorrscan.plugins.console_scripts',
            'rt_eqcorrscan.helpers',
            'rt_eqcorrscan.plugins.picker',
            'rt_eqcorrscan.plugins.magnitudes',
            'rt_eqcorrscan.plugins.plotter',
            'rt_eqcorrscan.plugins.plotter.rcet_plots',
            'rt_eqcorrscan.plugins.relocation',
            'rt_eqcorrscan.plugins.relocation.dt_correlations',
        ]
    if os.path.isdir("build"):
        shutil.rmtree("build")
    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
