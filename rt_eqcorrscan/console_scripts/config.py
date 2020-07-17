#!/usr/bin/env python3
"""
Make a basic config file with defaults.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

from rt_eqcorrscan.config import Config


def run(outfile):
    config = Config()
    config.write(outfile)
    print(f"Written config file: {outfile}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a default config file to disk for later editing")

    parser.add_argument(
        "-o", "--outfile", type=str, help="File to write config to",
        default="rt-eqcorrscan-config.yml")

    args = parser.parse_args()
    run(args.outfile)


if __name__ == "__main__":
    main()