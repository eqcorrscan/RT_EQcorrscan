RT-EQcorrscan API
=================

RT-EQcorrscan has a few submodules that add up to the whole. Herein you will
find the API for interacting with these modules.

Config
------

Methods for handling configuration of RT-EQcorrscan. These are used for the
installed scripts, but nothing else.

.. toctree::
   :maxdepth: 1

   submodules/config.config

Core
----

The heart of RT-EQcorrscan.

.. toctree::
   :maxdepth: 1

   submodules/core.database_manager
   submodules/core.reactor
   submodules/core.rt_match_filter

Plotting
--------

Real-time visualisation of the real-time matched-filter process.

.. toctree::
   :maxdepth: 1

   submodules/plotting.plot_buffer

Utils
-----

Utility modules for RT-EQcorrscan

.. toctree::
   :maxdepth: 1

   submodules/utils.event_trigger
   submodules/utils.seedlink