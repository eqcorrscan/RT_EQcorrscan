RT-EQcorrscan tutorials
=======================

RT-EQcorrscan is also ships with two
scripts: `real-time-mf` and `mf-reactor` which provide a command-line
interface for standard operations.  You will likely find that these scripts suffice
for most use-cases.

To use these scripts you will need to set up your configuration-file and build you
TemplateBank. See the tutorials (Class Interfaces) below for how to use TemplateBanks, and
how to configure RT-EQcorrscan.

Once you have built a TemplateBank, and set up your configuration file you
will be able to run the scripts with the `--config` flag. If you do not provide a
configuration file then the default configurations will be used.

real-time-mf
------------
The purpose of this script is to run a matched-filter detection in real-time for
a given region.  Run `real-time-mf --help` for details on the possible arguments.

To read in a set of templates for a given region and run a matched-filter using
these you would run:

.. code-block:: Bash
    real-time-mf --latitude -45 --longitude 45 --radius 2
    # All arguments here are in degrees

This would read in all template within this region.  If you want to only use
templates between specific dates you can use the `--template-starttime` and 
`--template-endtime` arguments.

If you want to detect events related to a particular event you can pass an event-id.
From the location and magnitude of this event a circular region of templates will
be read in and used:

.. code-block:: Bash
    real-time-mf --eventid 2016p858000  # This is GeoNet's Kaikoura event-id

mf-reactor
----------
The purpose of this script is to start a reactor process.  This will listen to
an FDSN service for events.  When new events are registered by the FDSN service
they are added to the template database (if they fall within the region defined
in the configuration file).  If a new event occurs that exceeds the triggering
thresholds (either magnitude, or event-rate within a region) then a RealTimeTribe
will be created using events from the TemplateBank.  The RealTimeTribe is
then used for a real-time matched-filter process. Additional events can trigger
the system and generate new RealTimeTribes.  Usually only the `--config <configuration-file>`
parameter needs to be specified.


Class Interfaces
----------------

The notebooks below provide a brief overview of some of the functionality
of the main classes in RT-EQcorrscan.

.. toctree::
    :maxdepth: 1

    examples/template_bank_tutorial.ipynb
    examples/catalog_listener_tutorial.ipynb
    examples/configuration_tutorial.ipynb
    examples/real_time_tribe_tutorial.ipynb
