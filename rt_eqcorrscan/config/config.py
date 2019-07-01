"""
Handle configuration of RT_EQcorrscan using a yaml file.
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


class ConfigAttribDict(AttribDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_yaml_dict(self):
        return {
            key.replace("_", " "): value
            for key, value in self.__dict__.items()}


class RTMatchFilterConfig(ConfigAttribDict):
    defaults = {
        "client": "GEONET",
        "client_type": "FDSN",
        "seedlink_server_url": "link.geonet.org.nz",
        "n_stations": 10,
        "max_distance": 1000,
        "buffer_capacity": 300,
        "detect_interval": 120,
        "plot": True,
        "threshold": .5,
        "threshold_type": "av_chan_corr",
        "trig_int": 2.0,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_client(self):
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


class ReactorConfig(ConfigAttribDict):
    defaults = {
        "magnitude_threshold": 6.0,
        "rate_threshold": 20.0,
        "rate_radius": 0.5,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlotConfig(ConfigAttribDict):
    defaults = {
        "plot_length": 600,
        "low_cut": 1.0,
        "high_cut": 10.0,
    }
    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DatabaseManagerConfig(ConfigAttribDict):
    defaults = {
        "event_path": ".",
        "event_format": "QUAKEML",
        "path_structure": "{year}/{month}/{event_id_short}",
        "event_ext": ".xml",
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
                "database_manager=...")

    def write(self, config_file):
        with open(config_file, "w") as f:
            f.write(dump(self.to_yaml_dict(), Dumper=Dumper))

    def to_yaml_dict(self):
        """ Make a more human readable yaml format """
        return {
            key.replace("_", " "): value.to_yaml_dict()
            for key, value in self.__dict__.items()}

    def setup_logging(self, **kwargs):
        logging.basicConfig(
            level=self.log_level, format=self.log_formatter, **kwargs)


def read_config(config_file=None):
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
