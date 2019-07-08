"""
RT_EQcorrscan: Near real-time matched-filtering for earthquake detection

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

__version__ = '0.0.1a'


from rt_eqcorrscan.rt_match_filter import RealTimeTribe
from rt_eqcorrscan.config.config import Config
from rt_eqcorrscan.database.database_manager import TemplateBank
from rt_eqcorrscan.reactor.reactor import Reactor
