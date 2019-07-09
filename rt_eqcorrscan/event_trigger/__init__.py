"""
Triggering functions for RT-EQcorrscan.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

from .triggers import (
    magnitude_rate_trigger_func, get_nearby_events, inter_event_distance,
    average_rate)
from .catalog_listener import filter_events, CatalogListener
from .listener import event_time
