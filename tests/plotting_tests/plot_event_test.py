"""
Test for event plotting in RT-eqcorrscan.
"""

import unittest
import pytest

from matplotlib.pyplot import Figure

from obspy.core.event import Event
from obspy.clients.fdsn import Client

from rt_eqcorrscan.plotting.plot_event import plot_event, _get_plot_starttime


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

    @pytest.mark.mpl_image_compare(baseline_dir="image_baseline")
    def test_reusing_figure(self):
        fig = plot_event(event=self.event, st=self.st, show=False,
                         fig=Figure())
        return fig

    def test_get_plot_time(self):
        origin_time = _get_plot_starttime(self.event, self.st)
        self.assertEqual(self.event.preferred_origin().time, origin_time)
        no_origins = Event(picks=self.event.picks)
        first_pick_time = _get_plot_starttime(no_origins, self.st)
        self.assertEqual(first_pick_time + 5,
                         min([pick.time for pick in no_origins.picks]))
        stream_starttime = _get_plot_starttime(Event(), self.st)
        self.assertEqual(stream_starttime,
                         min(tr.stats.starttime for tr in self.st))


if __name__ == "__main__":
    unittest.main()
