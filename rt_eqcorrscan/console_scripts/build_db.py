#!/usr/bin/env python3
"""
Build a template database based on configuration parameters.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import logging
import faulthandler

faulthandler.enable()

from obspy import UTCDateTime
from concurrent.futures import ProcessPoolExecutor

from rt_eqcorrscan.config import read_config
from rt_eqcorrscan.database import TemplateBank


Logger = logging.getLogger(__name__)


def run(
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    chunk_size: float = 30,
    rebuild: bool = True,
    max_workers: int = None,
    **kwargs
):
    config = read_config(config_file=kwargs.get("config_file", None))
    debug = kwargs.get("debug", False)
    if debug:
        config.log_level = "DEBUG"
        print(f"Using the following configuration:\n{config}")
    config.setup_logging()
    Logger.debug("Running in debug mode - expect lots of output!")

    client = config.rt_match_filter.get_client()

    template_bank = TemplateBank(
        config.database_manager.event_path,
        name_structure=config.database_manager.name_structure,
        event_format=config.database_manager.event_format,
        path_structure=config.database_manager.path_structure,
        event_ext=config.database_manager.event_ext,
        executor=ProcessPoolExecutor(max_workers=max_workers))

    # Break request into chunks
    if chunk_size < (endtime - starttime) / 86400:
        n_chunks = (endtime - starttime) / (chunk_size * 86400)
        Logger.info(f"Breaking into {n_chunks}")
        chunks = [(starttime + (i * chunk_size * 86400),
                   starttime + ((i + 1) * chunk_size * 86400))
                  for i in range(int(n_chunks))]
        if n_chunks > int(n_chunks):
            chunks.append((chunks[-1][-1], endtime))
    else:
        chunks = [(starttime, endtime)]

    for chunk_start, chunk_end in chunks:
        Logger.info(f"Downloading catalog between {chunk_start} and {chunk_end}")
        catalog = client.get_events(
            starttime=chunk_start, endtime=chunk_end,
            **config.reactor.catalog_lookup_kwargs)
        Logger.info(f"Will make templates for {len(catalog)} events")

        tribe = template_bank.make_templates(
            catalog=catalog, rebuild=rebuild, client=client, **config.template)
        Logger.info(f"Made {len(tribe)} templates")


def main():
    import argparse

    kwargs = {}
    parser = argparse.ArgumentParser(
        description="Build a TemplateBank; by default, if the "
                    "TemplateBank exists, only new templates will be added. "
                    "Use '-r' flag to enforce re-construction of templates "
                    "already in the TemplateBank")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file",
        required=False)
    parser.add_argument(
        "--debug", action="store_true", help="Flag to set log level to debug")
    parser.add_argument(
        "-s", "--starttime", type=str, default="1970-01-01",
        help="Starttime parsable by obspy's UTCDateTime to begin database from")
    parser.add_argument(
        "-e", "--endtime", type=str, default=None,
        help="Endtime parsable by obspy's UTCDateTime to end database at. "
             "Defaults to now")
    parser.add_argument(
        "-i", "--chunk-interval", type=float, default=30,
        help="Chunk size in DAYS for downloading and building database")
    parser.add_argument(
        "-r", "--rebuild", action="store_true",
        help="Force templates already in the database to be re-constructed")
    parser.add_argument(
        "-n", "--max-workers", type=int, default=None,
        help="Maximum workers for ProcessPoolExecutor, defaults to the number "
             "of cores on the machine")

    args = parser.parse_args()

    try:
        starttime = UTCDateTime(args.starttime)
    except Exception as e:
        Logger.error(e)
        raise NotImplementedError(
            f"Could not parse {args.starttime} to UTCDateTime")

    if args.endtime:
        try:
            endtime = UTCDateTime(args.endtime)
        except Exception as e:
            Logger.error(e)
            raise NotImplementedError(
                f"Could not parse {args.endtime} to UTCDateTime")
    else:
        endtime = UTCDateTime()

    kwargs.update({"debug": args.debug, "config_file": args.config})
    run(starttime=starttime, endtime=endtime, 
        chunk_size=args.chunk_interval, rebuild=args.rebuild,
        max_workers=args.max_workers, **kwargs)


if __name__ == "__main__":
    main()