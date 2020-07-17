RT-EQcorrscan installation
==========================

RT-EQcorrscan only supports Python versions >= 3.6. It is tested on Ubuntu, MacOS and Windows.

From conda-forge (Recommended!)
-------------------------------

If you do not have conda installed then see the conda `installation documentation here <https://docs.conda.io/en/latest/miniconda.html>`_.

Option 1:
.........

If you have an environment that you want to use you can try installing RT-EQcorrscan into that env:

.. code-block:: bash

    conda install -c conda-forge RT-EQcorrscan

Option 2:
.........

If you run into dependency resolution clashes, or you do not have an environment that you want to
install RT-EQcorrscan into then you can make a new environment with RT-EQcorrscan using:

.. code-block:: bash

    conda create -n RT-EQcorrscan -c conda-forge RT_EQcorrscan

From PyPi
---------

.. code-block:: bash

    pip install RT-EQcorrscan

From Source
-----------

First you will need to clone the RT-EQcorrscan repository:

.. code-block:: bash

    git clone https://github.com/eqcorrscan/RT_EQcorrscan.git
    cd RT-EQcorrscan

Then install the package using pip:

.. code-block:: bash

    pip install .

or if you want to change things you can install in *development* mode:

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py develop

or if you prefer using conda (note you may need to add the *conda-forge* channel to your channels list):

.. code-block:: bash

    conda install --file requirements.txt
    python setup.py install # Replace install with develop if you want development mode.