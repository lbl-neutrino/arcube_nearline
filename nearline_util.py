#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
import re


TIMEZONES = {
    'CST': '-0600',
    'CDT': '-0500',
    'CET': '+0100',
}


def date_from_filename(filename):
    basename = Path(filename).name
    match = re.search(r'\d\d\d\d_\d\d_\d\d_\d\d_\d\d_\d\d_[^.]+', basename)
    date_str = match.group()
    timezone = date_str.split('_')[-1]
    offset = TIMEZONES[timezone]
    date_str = date_str.replace(timezone, offset)
    return datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S_%z")
