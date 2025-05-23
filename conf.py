"""Configuration functionality."""

import argparse
import copy
import logging
import os
import platform
import textwrap
from pathlib import Path, PurePath

import yaml
from schema import And, Schema, SchemaError, Use


def get_local_computer_info() -> str:
    """Get identifying information about the current computer."""
    out = platform.node()
    if "SLURM_JOB_ID" in os.environ:
        out += "-" + os.environ["SLURM_JOB_ID"]
    return out


Path("logs").mkdir(parents=True, exist_ok=True)
LOG_LEVEL = logging.INFO
logging.basicConfig(
    filename=PurePath("logs") / f"{get_local_computer_info()}-burn.log",
    filemode="a",
    encoding="utf-8",
    level=LOG_LEVEL,
    format="%(levelname)s:%(asctime)s:%(funcName)s: %(message)s",
)
LOG = logging.getLogger("burn")

DEFAULT_OUTPUT_FILE_PATH = PurePath("results.csv")

BLANK_CONFIG: dict = {
    "computation": {
        "cpu": {
            "matrix_size": None,
            "replicates": None,
        },
        "gpu": {
            "matrix_size": None,
            "replicates": None,
        },
    },
}

DEFAULT_CONFIG_FILE_PATH = PurePath("config.yml")
DEFAULT_CONFIG = copy.deepcopy(BLANK_CONFIG)
DEFAULT_CONFIG["computation"]["cpu"]["matrix_size"] = 1000
DEFAULT_CONFIG["computation"]["cpu"]["replicates"] = 10
DEFAULT_CONFIG["computation"]["gpu"]["matrix_size"] = 10000
DEFAULT_CONFIG["computation"]["gpu"]["replicates"] = 10

POSITIVE_INTEGER = And(
    Use(int),  # type: ignore reportArgumentType
    lambda n: n > 0,
)
SCHEMA_RAW = copy.deepcopy(BLANK_CONFIG)
SCHEMA_RAW["computation"]["cpu"]["matrix_size"] = POSITIVE_INTEGER
SCHEMA_RAW["computation"]["cpu"]["replicates"] = POSITIVE_INTEGER
SCHEMA_RAW["computation"]["gpu"]["matrix_size"] = POSITIVE_INTEGER
SCHEMA_RAW["computation"]["gpu"]["replicates"] = POSITIVE_INTEGER
SCHEMA = Schema(SCHEMA_RAW)


def get_args() -> argparse.Namespace:
    """Get arguments from command line invocation."""
    parser = argparse.ArgumentParser(
        prog="burn",
        usage="burn.py ",
        description=textwrap.dedent(
            """
            Runs pytorch matmul on square matrices of size n, repeating k times,
            for each CPU and each GPU detected on the system. CPU runs are
            concurrent with each other, and GPU runs are concurrent with each
            other. CPU and GPU runs are not concurrent.

            No parameters are required, looks for config.yml in the working
            directory by default.
            """,
        ).strip(),
    )
    parser.add_argument(
        "-c",
        "--config-file",
        metavar="configfile",
        nargs="?",
        type=PurePath,
        default=DEFAULT_CONFIG_FILE_PATH,
        help="Path to config file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="outputfile",
        nargs="?",
        type=PurePath,
        default=DEFAULT_OUTPUT_FILE_PATH,
        help="Desired path to output CSV file.",
    )

    return parser.parse_args()


def load_or_build_config(_file: PurePath) -> dict:
    """Load or build config."""
    try:
        with Path(_file).open("r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        LOG.exception(
            "unable to open config at %s, creating default and exiting",
            str(_file),
        )
        with Path(DEFAULT_CONFIG_FILE_PATH).open("w") as f:
            yaml.dump(DEFAULT_CONFIG, f)
        raise

    try:
        validated = SCHEMA.validate(config)
    except SchemaError:
        LOG.exception("format config like the following")
        LOG.exception(yaml.dump(DEFAULT_CONFIG))
        raise

    return validated
