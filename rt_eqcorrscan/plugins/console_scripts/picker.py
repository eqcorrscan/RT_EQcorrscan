"""
Console script entry point for lag-calc plugin.
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.picker.picker_runner import Picker


Logger = logging.getLogger("lag-calc-plugin")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lag Calc plugin")

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=True)
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO")
    parser.add_argument(
        "--log-formatter", type=str,
        default="%(asctime)s\t[%(processName)s:%(threadName)s]: " \
                "%(name)s\t%(levelname)s\t%(message)s")
    parser.add_argument(
        "--log-file", type=str, default="lag-calc.log")
    parser.add_argument(
        "--log-to-screen", "-s", action="store_true")
    parser.add_argument(
        "--simulation", action="store_true",
        help="Flag to notify if this is a simulation - extra output will be"
             "provided in simulation mode.")

    args = parser.parse_args()

    _setup_logging(
        log_level=args.log_level, log_formatter=args.log_formatter,
        screen=args.log_to_screen, file=True, filename=args.log_file)

    picker = Picker(config_file=args.config)
    if args.simulation:
        picker._write_sim_catalogues = True
    picker.run()


if __name__ == "__main__":
    main()