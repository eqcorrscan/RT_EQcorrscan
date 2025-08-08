"""
Console entry point for plotter runner
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.plotter.plotter_runner import Plotter


Logger = logging.getLogger("plotter-plugin")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plotter Plugin")

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
        "--log-file", type=str, default="plotter.log")
    parser.add_argument(
        "--log-to-screen", "-s", action="store_true")
    parser.add_argument(
        "--now", action="store_false",
        help="Make a plot now and stop. Do not continue watching for new events"
    )
    parser.add_argument(
        "--simulation", action="store_true",
        help="Flag to notify if this is a simulation - extra output will be"
             "provided in simulation mode.")
    parser.add_argument(
        "--simulation-time-offset", type=float, default=0.0,
        help="Seconds to subtract from now to give simulated time")

    args = parser.parse_args()

    _setup_logging(
        log_level=args.log_level, log_formatter=args.log_formatter,
        screen=args.log_to_screen, file=True, filename=args.log_file)

    plotter = Plotter(config_file=args.config)
    plotter.simulation_time_offset = args.simulation_time_offset
    if args.simulation:
        Logger.info("Nothing to be done for simulation for picker")
    plotter.run(loop=args.now)


if __name__ == "__main__":
    main()
