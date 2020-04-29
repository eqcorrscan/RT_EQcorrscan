"""
Utilities for emulating obspy clients using local data. Relies on obsplus
"""

import logging

from typing import Union
from obspy.clients.fdsn import Client
from obsplus.bank import WaveBank, EventBank, StationBank


Logger = logging.getLogger(__name__)


class ClientBank(object):
    """
    Thin routing wrapper for obsplus Banks to act as a client.

    Parameters
    ----------
    wave_bank
        WaveBank with seismic data
    event_bank
        EventBank with event data
    station_bank
        StationBank with station data. Note that as of 16/07/2019 StationBank
        was incomplete

    Notes
    -----
        All attributes can be substituted for different (or the same) client.
    """
    def __init__(
        self,
        wave_bank: Union[Client, WaveBank] = None,
        event_bank: Union[Client, EventBank] = None,
        station_bank: Union[Client, StationBank] = None,
    ):
        self.wave_bank = wave_bank
        self.station_bank = station_bank
        self.event_bank = event_bank
        self.base_url = "I'm not a real client!"

    def get_stations(self, *args, **kwargs):
        if self.station_bank is None:
            raise NotImplementedError("No station_bank provided")
        return self.station_bank.get_stations(*args, **kwargs)

    def get_stations_bulk(self, *args, **kwargs):
        if self.station_bank is None:
            raise NotImplementedError("No station_bank provided")
        return self.station_bank.get_stations_bulk(*args, **kwargs)

    def get_waveforms(self, *args, **kwargs):
        if self.wave_bank is None:
            raise NotImplementedError("No wave_bank provided")
        return self.wave_bank.get_waveforms(*args, **kwargs)

    def get_waveforms_bulk(self, *args, **kwargs):
        if self.wave_bank is None:
            raise NotImplementedError("No wave_bank provided")
        return self.wave_bank.get_waveforms_bulk(*args, **kwargs)

    def get_events(self, *args, **kwargs):
        if self.event_bank is None:
            raise NotImplementedError("No event_bank provided")
        return self.event_bank.get_events(*args, **kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
