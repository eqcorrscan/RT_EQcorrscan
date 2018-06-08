"""
Example of real-time plotting using RT_EQcorrscan
"""
import time

from rt_eqcorrscan.utils.seedlink import RealTimeClient

rt_client = RealTimeClient(
    server_url="link.geonet.org.nz", buffer_capacity=60,
    log_level='warning')
rt_client.select_stream(net="NZ", station="FOZ", selector="HH?")
rt_client.select_stream(net="NZ", station="WVZ", selector="HH?")
rt_client.select_stream(net="NZ", station="JCZ", selector="HH?")
rt_client.select_stream(net="NZ", station="GCSZ", selector="EH?")

rt_client.background_run(plot=True, plot_length=600, ylimits=(-5, 5), 
                         size=(16, 12))
time.sleep(86400) # Sleep for a day
rt_client.background_stop()