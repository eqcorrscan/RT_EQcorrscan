RT-EQcorrscan installation
==========================

RT-EQcorrscan is only supported for Python versions >= 3.6. Currently
RT-EQcorrscan is only available from source on github. We do plan on releasing full
releases on PyPi and conda-forge, once the project has stabilised a little more.

From conda-forge
----------------

Not yet implemented.

From PyPi
---------

Not yet implemented.

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