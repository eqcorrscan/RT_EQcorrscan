"""
Handle configuration of RT_EQcorrscan using a yaml file.
"""

DEFAULTS = {
    # Set defaults here
    "client": "GEONET",  # TODO: Fill out
}

REQUIRED = {
    "threshold", "threshold_type",  # TODO: Fill out
}


class Config(object):
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = "default_config.yml"
        self.config_file = config_file
        # Set default attributes
        # Load from file
        # Check that required are there
        # Update attributes
        # Convert client_type to an obspy Client


if __name__ == "__main__":
    import doctest

    doctest.testmod()
