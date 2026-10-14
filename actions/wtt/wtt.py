#!/usr/bin/env python3
"""
Nearline whole-tile-trigger (WTT) monitor: tag WTTs in one flow file and add
a per-file summary to a shared JSON.
Re-processing a file replaces its entry instead of duplicating it.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

import h5py
from flufl.lock import Lock

import wtt_funcs as WTT
from nearline_util import date_from_filename


def main(input_file, output_file_json, **cuts):
    print('Opening', input_file)
    with h5py.File(input_file, 'r') as f:
        result = WTT.tag_file(f, cuts={k: v for k, v in cuts.items() if v is not None})

    try:
        timestamp = date_from_filename(input_file)
    except (AttributeError, KeyError):    # no date in the name: use the first event
        timestamp = datetime.fromtimestamp(result['first_unix_ts'], tz=timezone.utc)

    tiles = result['tagged_tiles']
    wtt_events = {t['event_id'] for t in tiles}
    wtt_beam_events = {t['event_id'] for t in tiles if t['is_beam']}
    run_time_s = None
    if result['first_unix_ts'] is not None:
        run_time_s = max(result['last_unix_ts'] - result['first_unix_ts'], 1)

    entry = dict(timestamp=timestamp.isoformat(),
                 file=os.path.basename(input_file),
                 run_time_s=run_time_s,
                 n_events=result['n_events'],
                 n_beam_events=result['n_beam_events'],
                 n_wtt_events=len(wtt_events),
                 n_wtt_beam_events=len(wtt_beam_events),
                 n_wtt_tiles=len(tiles),
                 n_candidate_tiles=result['n_candidate_tiles'],
                 n_step_tiles=result['n_step_tiles'],
                 wtt_tiles_by_io_group={str(k): v for k, v in sorted(Counter(t['io_group'] for t in tiles).items())},
                 wtt_tiles=tiles,
                 cuts=dict(WTT.DEFAULT_CUTS, **{k: v for k, v in cuts.items() if v is not None}))
    print("Timestamp: %s, events: %d (beam %d), WTT events: %d (beam %d), WTT tiles by IO group: %s"
          % (entry['timestamp'], entry['n_events'], entry['n_beam_events'], entry['n_wtt_events'],
             entry['n_wtt_beam_events'], entry['wtt_tiles_by_io_group']))

    # same locking as actions/lifetime: several workers may finish at once
    with Lock(output_file_json + '.lock'):
        data = {"wtt": []}
        if os.path.exists(output_file_json):
            with open(output_file_json) as fin:
                data = json.load(fin)
        data["wtt"] = [e for e in data["wtt"] if e['file'] != entry['file']] + [entry]
        with open(output_file_json, "w") as fout:
            json.dump(data, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_file', required=True, help="Path to charge flow hdf5 file")
    parser.add_argument('--output_file_json', required=True, help="Shared WTT summary json")
    for name, default in WTT.DEFAULT_CUTS.items():
        parser.add_argument('--' + name, type=type(default), default=None,
                            help='default: %s' % default)
    args = parser.parse_args()
    print(args)
    main(**vars(args))
