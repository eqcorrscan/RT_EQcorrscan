# RT EQcorrscan
## Real-time implementation of EQcorrscan's matched-filter earthquake detection

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![test](https://github.com/eqcorrscan/RT_EQcorrscan/workflows/test/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/eqcorrscan/RT_EQcorrscan/branch/master/graph/badge.svg)](https://codecov.io/gh/eqcorrscan/RT_EQcorrscan)
[![Documentation Status](https://readthedocs.org/projects/rt-eqcorrscan/badge/?version=latest)](https://rt-eqcorrscan.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)  

# Installation

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/rt-eqcorrscan/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)

Either install EQcorrscan from conda-forge (recommended) using something like:
```bash
conda install -c conda-forge RT-EQcorrscan
```

or from pypi using something like:
```bash
pip install RT-EQcorrscan
```

or from source by cloning this repository and running pip install:
```bash
git clone https://github.com/eqcorrscan/RT_EQcorrscan.git
cd RT_EQcorrscan
pip install .  # This should install the required dependencies.
```

# Usage

Have a peruse of the [docs](https://rt-eqcorrscan.readthedocs.io/en/latest/)
to see more information on how to use RT-EQcorrscan. 

Feel free to edit the source code and add/change how RT-EQcorrscan works to make it fit
your application.  If you find any bugs, or have a feature that you want to add, then
please do! It would be really valuable to make this project better! See the section 
below on contributing.

# Contributing

RT_EQcorrscan is ready to play with, but don't expect it to be stable yet. If
you have any contributions they would be appreciated! Please fork the repository
and create a pull-request on Master.

# Funding

The creation of this project was funded by the Earthquake Commission of New Zealand (EQC).
Currently the maintainence of this project is unfunded.

# Citation

If you use our software in your research please cite 
[our paper on the RT-EQcorrscan package](https://pubs.geoscienceworld.org/ssa/srl/article/doi/10.1785/0220200171/590814/RT-EQcorrscan-Near-Real-Time-Matched-Filtering-for).
These citations help to keep the developers in work and keep maintaining these software!

> Chamberlain, C. J., J. Townend, and M. C. Gerstenberger (2020). 
> RT-EQcorrscan: Near-Real-Time Matched-Filtering for Rapid Development
> of Dense Earthquake Catalogs, Seismol. Res. Lett. XX, 1–11, 
> doi: 10.1785/0220200171.
