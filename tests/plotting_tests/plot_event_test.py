"""
Test for event plotting in RT-eqcorrscan.
"""

import unittest
import pytest

from obspy.clients.fdsn import Client

from rt_eqcorrscan.plotting.plot_event import plot_event


class EventPlottingMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        client = Client("GEONET")
        cls.event = client.get_events(eventid="2019p428761")[0]
        # First eight picked stations
        stations = {'HDWS', 'JCZ', 'MECS', 'MLZ', 'MSZ', 'NSBS', 'QTPS', 'WKZ'}
        bulk = [
            (p.waveform_id.network_code, p.waveform_id.station_code,
             p.waveform_id.location_code, p.waveform_id.channel_code,
             p.time - 10, p.time + 20)
            for p in cls.event.picks if p.phase_hint == 'P'
            and p.waveform_id.station_code in stations]
        cls.st = client.get_waveforms_bulk(bulk)

    @pytest.mark.mpl_image_compare(baseline_dir="image_baseline")
    def test_event_plotting(self):
        fig = plot_event(event=self.event, st=self.st, show=False)
        return fig


if __name__ == "__main__":
    unittest.main()