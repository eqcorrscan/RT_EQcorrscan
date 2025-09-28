"""
Console entry point for growclust runner.
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.relocation.growclust_runner import GrowClust


Logger = logging.getLogger("growclust-plugin")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GrowClust Plugin")

    parser.add_argument(
        "--config", "-c", type=str,
        help="Path to configuration file", required=True)
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO")
    parser.add_argument(
        "--log-formatter", type=str,
        default="%(asctime)s\t[%(processName)s:%(threadName)s]: " \
                "%(name)s\t%(levelname)s\t%(message)s")
    parser.add_argument(
        "--log-file", type=str, default="growclust.log")
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

    growclust = GrowClust(config_file=args.config)
    if args.simulation:
        Logger.info("Setting simulation to True")
        growclust._write_sim_catalogues = True
    growclust.run()


if __name__ == "__main__":
    main()
