"""
RT_EQcorrscan: Near real-time matched-filtering for earthquake detection
"""

__version__ = '0.0.2'


from rt_eqcorrscan.rt_match_filter import RealTimeTribe
from rt_eqcorrscan.config.config import Config, read_config
from rt_eqcorrscan.database.database_manager import TemplateBank
from rt_eqcorrscan.reactor.reactor import Reactor
