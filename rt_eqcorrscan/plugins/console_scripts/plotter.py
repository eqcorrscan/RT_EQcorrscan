"""
Console entry point for plotter runner
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.plotter.plotter_runner import main as plotter_runner


Logger = logging.getLogger("hyp-plugin")


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

    args = parser.parse_args()

    _setup_logging(
        log_level=args.log_level, log_formatter=args.log_formatter,
        screen=args.log_to_screen, file=True, filename=args.log_file)

    plotter_runner(config_file=args.config, loop=args.now)


if __name__ == "__main__":
    main()
