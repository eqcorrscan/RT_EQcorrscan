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

To start with, you will need the current development version of
`ObsPlus <https://github.com/niosh-mining/obsplus>`_. A full release
should be coming soon, so if you are looking at these docs after February 2020,
check whether obsplus 0.1.0 has been released yet.  If it has, skip this step!
Otherwise, if we haven't finished packaging obsplus, you will need to
clone the obsplus repository:

.. code-block:: bash

    git clone https://github.com/niosh-mining/obsplus/obsplus.git

Then install obsplus from source:

.. code-block:: bash

    cd obsplus
    pip install .

Now you can proceed to installing RT-EQcorrscan. First you will need to clone
the RT-EQcorrscan repository:

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