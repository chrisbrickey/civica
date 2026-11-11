"""Reading log lines written in the logging convention: an event name, then key=value fields."""

import logging
import re
from collections.abc import Iterable
from typing import Final

# An event name, then single-token key=value fields separated by single spaces.
LINE_FORMAT: Final = re.compile(r"^[a-z_]+( [a-z_]+=\S+)+$")
# The turn_id of a line logged before the turn's id is known.
UNKNOWN_TURN_ID: Final = "unknown"

LogLine = tuple[str, dict[str, str]]


def parse_line(message: str) -> LogLine:
    """The event name and fields of one line, which must follow the logging convention."""
    assert LINE_FORMAT.match(message), f"not an event line with key=value fields: {message!r}"
    event, *pairs = message.split(" ")
    return event, {key: value for key, _, value in (pair.partition("=") for pair in pairs)}


def lines_from(records: Iterable[logging.LogRecord], logger: str) -> list[LogLine]:
    """Every line the named logger wrote, parsed."""
    return [parse_line(record.getMessage()) for record in records if record.name == logger]


def events(lines: Iterable[LogLine]) -> list[str]:
    return [event for event, _fields in lines]


def only(lines: Iterable[LogLine], event: str) -> dict[str, str]:
    """The fields of the one line logged for `event`."""
    matching = [fields for name, fields in lines if name == event]
    assert len(matching) == 1, f"expected one {event} line, found {len(matching)}"
    return matching[0]
