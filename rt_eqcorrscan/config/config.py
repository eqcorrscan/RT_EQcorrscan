"""
Handle configuration of RT_EQcorrscan using a yaml file.

    This file is part of rt_eqcorrscan.

    rt_eqcorrscan is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    rt_eqcorrscan is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with rt_eqcorrscan.  If not, see <https://www.gnu.org/licenses/>.

"""
import logging
import os

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from obspy.core.util import AttribDict


Logger = logging.getLogger(__name__)


class _ConfigAttribDict(AttribDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_yaml_dict(self):
        return {
            key.replace("_", " "): value
            for key, value in self.__dict__.items()}

    def __eq__(self, other):
        for key in self.__dict__.keys():
            if not hasattr(other, key):
                return False
            if self[key] != other[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class RTMatchFilterConfig(_ConfigAttribDict):
    """
    A holder for configuration values for real-time matched-filtering.

    Works like a dictionary and can have anything added to it.
    """
    defaults = {
        "client": "GEONET",
        "client_type": "FDSN",
        "seedlink_server_url": "link.geonet.org.nz",
        "n_stations": 10,
        "max_distance": 1000.,
        "buffer_capacity": 300.,
        "detect_interval": 120.,
        "plot": True,
        "threshold": .5,
        "threshold_type": "av_chan_corr",
        "trig_int": 2.0,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_client(self):
        """ Get the client instance given the set parameters. """
        from obspy import clients

        try:
            _client_module = clients.__getattribute__(self.client_type.lower())
        except AttributeError as e:
            Logger.error(e)
            return None
        try:
            client = _client_module.Client(self.client)
        except Exception as e:
            Logger.error(e)
            return None
        return client


class ReactorConfig(_ConfigAttribDict):
    """
    A holder for configuration values for the reactor.

    Works like a dictionary and can have anything added to it.
    """
    defaults = {
        "magnitude_threshold": 6.0,
        "rate_threshold": 20.0,
        "rate_radius": 0.5,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlotConfig(_ConfigAttribDict):
    """
    A holder for configuration values for real-time matched-filter plotting.

    Works like a dictionary and can have anything added to it.
    """
    defaults = {
        "plot_length": 600.,
        "low_cut": 1.0,
        "high_cut": 10.0,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DatabaseManagerConfig(_ConfigAttribDict):
    """
    A holder for configuration values for database management.

    Works like a dictionary and can have anything added to it.
    """
    defaults = {
        "event_path": ".",
        "event_format": "QUAKEML",
        "event_name_structure": "{event_id_end}",
        "path_structure": "{year}/{month}/{event_id_end}",
        "event_ext": ".xml",
        "min_stations": 5,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


KEY_MAPPER = {
    "rt_match_filter": RTMatchFilterConfig,
    "reactor": ReactorConfig,
    "plot": PlotConfig,
    "database_manager": DatabaseManagerConfig,
}


class Config(object):
    """
    Base configuration parameters from RT_EQcorrscan.

    Parameters
    ----------
    log_level
        Any parsable string for logging.basicConfig
    log_formatter
        Any parsable string formatter for logging.basicConfig
    rt_match_filter
        Config values for real-time matchec-filtering
    reactor
        Config values for the Reactor
    plot
        Config values for real-time plotting
    database_manager
        Config values for the database manager.
    """
    log_level = "INFO"
    log_formatter = "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
    rt_match_filter = RTMatchFilterConfig()
    reactor = ReactorConfig()
    plot = PlotConfig()
    database_manager = DatabaseManagerConfig()

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in KEY_MAPPER.keys():
                raise NotImplementedError(
                    "Unsupported argument type: {0}".format(key))
            if isinstance(value, dict):
                self.__dict__[key] = KEY_MAPPER[key](value)
            else:
                assert isinstance(value, type(self.__dict__[key]))
                self.__dict__[key] = value

    def __repr__(self):
        return ("Config(rt_match_filter=..., reactor=..., plot=..., "
                "database_manager=...)")

    def __eq__(self, other):
        if not isinstance(other, Config):
            return False
        for key in self.__dict__.keys():
            if not hasattr(other, key):
                return False
            if not self.__dict__[key] == other.__dict__[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def write(self, config_file: str) -> None:
        """
        Write the configuration to a tml formatted file.

        Parameters
        ----------
        config_file
            path to the configuration file. Will overwrite and not warm
        """
        with open(config_file, "w") as f:
            f.write(dump(self.to_yaml_dict(), Dumper=Dumper))

    def to_yaml_dict(self) -> dict:
        """ Make a more human readable yaml format """
        return {
            key.replace("_", " "): value.to_yaml_dict()
            for key, value in self.__dict__.items()}

    def setup_logging(self, **kwargs):
        """Set up logging using the logging parameters."""
        logging.basicConfig(
            level=self.log_level, format=self.log_formatter, **kwargs)


def read_config(config_file=None) -> Config:
    """
    Read configuration from a yml file.

    Parameters
    ----------
    config_file
        path to the configuration file.

    Returns
    -------
    Configuration with required defaults filled and updated based on the
    contents of the file.
    """
    if config_file is None:
        return Config()
    if not os.path.isfile(config_file):
        raise FileNotFoundError(config_file)
    with open(config_file, "rb") as f:
        configuration = load(f, Loader=Loader)
    config_dict = {}
    for key, value in configuration.items():
        config_dict.update(
            {key.replace(" ", "_"):
                 {_key.replace(" ", "_"): _value
                  for _key, _value in value.items()}})
    return Config(**config_dict)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
