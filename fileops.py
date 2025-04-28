"""Contains code to write results to disk."""

from pathlib import Path, PurePath

from stats import Summary


def to_csv(filepath: PurePath, results: dict[str, Summary]) -> None:
    """Write results to CSV file at filepath."""
    if not results:
        return

    devices_sorted = sorted(results.keys())
    stats = {device: results[device].to_dict() for device in devices_sorted}

    stat_names: set[str] = set()
    for stat in stats.values():
        stat_names |= stat.keys()
    stat_names_sorted = sorted(stat_names)

    header_line = ",".join(["device", *stat_names_sorted])
    lines = [header_line]
    for device, stat in stats.items():
        values_sorted = [stat[name] for name in stat_names_sorted]
        values_to_write = [f"{value:.17f}" for value in values_sorted]
        stat_line = ",".join([device, *values_to_write])
        lines.append(stat_line)

    Path(filepath.parent).mkdir(parents=True, exist_ok=True)
    with Path(filepath).open("w") as f:
        f.write("\n".join(lines))
