"""
Console entry point for hyp runner.
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.relocation.hyp_runner import Hyp


Logger = logging.getLogger("hyp-plugin")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hyp Plugin")

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
        "--log-file", type=str, default="hyp.log")
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

    hyp = Hyp(config_file=args.config)
    if args.simulation:
        hyp._write_sim_catalogues = True
    hyp.run()


if __name__ == "__main__":
    main()
