RT-EQcorrscan tutorials
=======================

RT-EQcorrscan is designed to handle most workflows via command-line scripts.
These scripts are:

- `rteqcorrscan-build-db`_
- `rteqcorrscan-config`_
- `rteqcorrscan-reactor`_
- `rteqcorrscan-real-time-match`_
- `rteqcorrscan-simulation`_

If you find that these scripts do not meet your use-case, you have full power
to write your own scripts that interact directly with the API

Command-line Interfaces (Scripts)
---------------------------------

To set-up your system for real-time matched-filtering you will need to
first generate a config file. The `rteqcorrscan-config`_ script will
generate a config file with default values that you can adjust.

**Need to document these values**

Once you have a config file that you are happy with you will need to generate
a template database. Building a template database can be quite a slow process,
especially for long datasets!  The `rteqcorrscan-build-db`_ provides a simple
way to start building your database from scratch.  If you already have templates
and/or event files then you can interface with the `Template-database`_ API
directly.

Once you have a configuration file and database you are ready to go! You can run
the `rteqcorrscan-reactor`_ script to start the full RT-EQcorrscan system, which
will listen to your chosen event service and react to any large earthquakes or
high-rate sequences (as defined by your config file).

Once one of these events happens
a real-time matched-filter will be started using templates from your database. If your
event service finds more events in your region of interest, these will be added
to your running real-time matched-filter database and run through previous data
to *fill-in* past events.

Once the `Reactor` has started a real-time system, it keeps listening for other
possible triggers.  Because of this, a whole country can be covered at once
(we haven't run a global reactor, but you could give it a go!), and events that
occur in different places can be reacted to.

The final script provided is the `rteqcorrscan-simulation`_ script, which allows
you to test your system on past events.  This can be really handy to get an idea
of what detections you might expect, and lets you check that RT-EQcorrscan is working
as you expect it!

rteqcorrscan-config
^^^^^^^^^^^^^^^^^^^

**Configure your system.**

rteqcorrscan-build-db
^^^^^^^^^^^^^^^^^^^^^

**Build your template database.**

rteqcorrscan-reactor
^^^^^^^^^^^^^^^^^^^^

**Start a reactor process - if something happens, your system will react.**

rteqcorrscan-real-time-match
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Run a real-time matched-filter for a specific region or event.**

rteqcorrscan-simulation
^^^^^^^^^^^^^^^^^^^^^^^

**Simulate a past period of interest: useful for testing!**


Class Interfaces
----------------

The notebooks below provide a brief overview of some of the functionality
of the main classes in RT-EQcorrscan.

.. toctree::
    :maxdepth: 1

    examples/configuration_tutorial.ipynb
    examples/template_bank_tutorial.ipynb
    examples/catalog_listener_tutorial.ipynb
    examples/real_time_tribe_tutorial.ipynb
