"""
Console entry point for hyp runner.
"""

import logging

from rt_eqcorrscan.config.config import _setup_logging
from rt_eqcorrscan.plugins.relocation.nll_runner import NLL


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
        "--log-file", type=str, default="nll.log")
    parser.add_argument(
        "--log-to-screen", "-s", action="store_true")

    args = parser.parse_args()

    _setup_logging(
        log_level=args.log_level, log_formatter=args.log_formatter,
        screen=args.log_to_screen, file=True, filename=args.log_file)

    nll = NLL(config_file=args.config)
    nll.run()


if __name__ == "__main__":
    main()