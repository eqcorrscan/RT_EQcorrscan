"""
Overarching tool for listening to and triggering from FDSN earthquakes.

TODO: Write this script: This should:
 - Use pre-computed Tribes covering patches of the country,
 - Listen to GeoNet earthquake feed
 - If an earthquake of interest happens, load the tribe for that region and
   start the real-time matched-filter
 - Once the detection rate drops low enough, stop running it?

 - Alongside this - check whether new detections made by GeoNet need to be
   included in the database.
"""


class Reactor(object):
    def __init__(self, client, listener, template_database):
        self.client = client
        self.listener = listener
        self.template_databse = template_database

    def run(self):
        self.listener.background_run()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
