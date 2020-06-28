RT-EQcorrscan tutorials
=======================

RT-EQcorrscan is designed to handle most workflows via command-line scripts.
These scripts are:

- `rteqcorrscan-build-db`_
- `rteqcorrscan-config`_
- `rteqcorrscan-reactor`_
- `rteqcorrscan-real-time-match`_
- `rteqcorrscan-simulation`_
- `rteqcorrscan-bench`_

If you find that these scripts do not meet your use-case, you have full power
to write your own scripts that interact directly with the :doc:`API <../api/index>`.

Command-line Interfaces (Scripts)
---------------------------------

To set-up your system for real-time matched-filtering you will need to
first generate a config file. The `rteqcorrscan-config`_ script will
generate a config file with default values that you can adjust.

Once you have a config file that you are happy with you will need to generate
a template database. Building a template database can be quite a slow process,
especially for long datasets!  The `rteqcorrscan-build-db`_ provides a simple
way to start building your database from scratch.  If you already have templates
and/or event files then you can interface with the :doc:`../api/modules/database` API
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

Once the **Reactor** has started a real-time system, it keeps listening for other
possible triggers.  Because of this, a whole country can be covered at once
(we haven't run a global reactor, but you could give it a go!), and events that
occur in different places can be reacted to.

The final script provided is the `rteqcorrscan-simulation`_ script, which allows
you to test your system on past events.  This can be really handy to get an idea
of what detections you might expect, and lets you check that RT-EQcorrscan is working
as you expect it!

Note that these docs are **not automatically updated** and may be out-of-date.
To confirm the arguments for your version of RT-EQcorrscan, use the :code:`--help` flag
of the scripts.

rteqcorrscan-config
^^^^^^^^^^^^^^^^^^^

**Configure your system.**

.. code-block:: bash

    usage: rteqcorrscan-config [-h] [-o OUTFILE]

    Write a default config file to disk for later editing

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTFILE, --outfile OUTFILE
                            File to write config to

To learn more about the possible configuration values and their defaults, check out
the :doc:`../api/modules/config` api documentation.

rteqcorrscan-build-db
^^^^^^^^^^^^^^^^^^^^^

**Build your template database.**

.. code-block:: bash

    usage: rteqcorrscan-build-db [-h] [--config CONFIG] [--debug] [-s STARTTIME]
                                 [-e ENDTIME] [-r] [-n MAX_WORKERS]

    Build a TemplateBank; by default, if the TemplateBank exists, only new
    templates will be added. Use '-r' flag to enforce re-construction of templates
    already in the TemplateBank

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG, -c CONFIG
                            Path to configuration file
      --debug               Flag to set log level to debug
      -s STARTTIME, --starttime STARTTIME
                            Starttime parsable by obspy's UTCDateTime to begin
                            database from
      -e ENDTIME, --endtime ENDTIME
                            Endtime parsable by obspy's UTCDateTime to end
                            database at
      -r, --rebuild         Force templates already in the database to be re-
                            constructed
      -n MAX_WORKERS, --max-workers MAX_WORKERS
                            Maximum workers for ProcessPoolExecutor, defaults to
                            the number of cores on the machine


rteqcorrscan-reactor
^^^^^^^^^^^^^^^^^^^^

**Start a reactor process - if something happens, your system will react.**

.. code-block:: bash

    usage: rteqcorrscan-reactor [-h] [--config CONFIG] [--debug] [-u]

    Run the RT_EQcorrscan Reactor

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG, -c CONFIG
                            Path to configuration file
      --debug               Flag to set log level to debug
      -u, --update-bank     Flag to update template bank index before running, use
                            if events have been manually added


rteqcorrscan-real-time-match
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Run a real-time matched-filter for a specific region or event.**

.. code-block:: bash

    usage: rteqcorrscan-real-time-match [-h] [--eventid EVENTID]
                                        [--latitude LATITUDE]
                                        [--longitude LONGITUDE] [--radius RADIUS]
                                        [--config CONFIG]
                                        [--template-starttime TEMPLATE_STARTTIME]
                                        [--template-endtime TEMPLATE_ENDTIME]
                                        [--starttime STARTTIME]
                                        [--speed-up SPEED_UP] [--debug]
                                        [--local-archive]

    Real Time Matched Filter

    optional arguments:
      -h, --help            show this help message and exit
      --eventid EVENTID, -e EVENTID
                            Triggering event ID
      --latitude LATITUDE   Latitude for template-search
      --longitude LONGITUDE
                            Longitude for template-search
      --radius RADIUS       Radius (in degrees) for template-search
      --config CONFIG, -c CONFIG
                            Path to configuration file
      --template-starttime TEMPLATE_STARTTIME
                            Start-time as UTCDateTime parsable string to collect
                            templates from
      --template-endtime TEMPLATE_ENDTIME
                            End-time as UTCDateTime parsable string to collect
                            templates up to.
      --starttime STARTTIME
                            Start-time for real-time simulation for past data
      --speed-up SPEED_UP   Speed-up factor for past data - unused for real-time
      --debug               Flag to set log level to debug
      --local-archive       Flag to use a local archive for waveform data, defined
                            in config file


rteqcorrscan-simulation
^^^^^^^^^^^^^^^^^^^^^^^

**Simulate a past period of interest: useful for testing!**

.. code-block:: bash

    usage: rteqcorrscan-simulation [-h] --quake QUAKE [--config CONFIG]
                                   [--db-duration DB_DURATION] [--radius RADIUS]
                                   --client CLIENT [--templates-made] [--debug]

    optional arguments:
      -h, --help            show this help message and exit
      --quake QUAKE         Earthquake to synthesise real-time, either the event-
                            id or a known key. Known events are: {'eketahuna':
                            '2014p051675', 'cook-strait': '2013p543824'}
      --config CONFIG, -c CONFIG
                            Path to configuration file
      --db-duration DB_DURATION
                            Number of days to generate the database for prior to
                            the chosen event
      --radius RADIUS       Radius in degrees to build database for
      --client CLIENT       Client to get data from, must have an FDSN waveform
                            and event service
      --templates-made      Flag to not make new templates - use if re-running an
                            old DB
      --debug               Flag to run in debug mode, with lots of output to
                            screen

rteqcorrscan-bench
^^^^^^^^^^^^^^^^^^

**Benchmark your configuration on your system**

Test your configuration and the limits of real-time detection on your system.
RTEQcorrscan runs detections in near-real-time, if it takes longer to make
detections than the length of the data detecting in, then the process will
fall behind.

.. code-block:: bash

    usage: rteqcorrscan-bench [-h] -t N_TEMPLATES [N_TEMPLATES ...] -n N_CHANNELS
                              [--config CONFIG]

    Benchmark rteqcorrscan for a range of templates

    optional arguments:
      -h, --help            show this help message and exit
      -t N_TEMPLATES [N_TEMPLATES ...], --n-templates N_TEMPLATES [N_TEMPLATES ...]
                            Sequence of the number of templates to run
      -n N_CHANNELS, --n-channels N_CHANNELS
                            Number of channels to run with
      --config CONFIG, -c CONFIG
                            Path to configuration file
      --verbose, -v         Print output from logging to screen



Class Interfaces
----------------

The notebooks below provide a brief overview of some of the functionality
of the main classes in RT-EQcorrscan.

.. toctree::
    :maxdepth: 1

    ../examples/configuration_tutorial.ipynb
    ../examples/template_bank_tutorial.ipynb
    ../examples/catalog_listener_tutorial.ipynb
    ../examples/real_time_tribe_tutorial.ipynb
