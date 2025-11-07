#!/usr/bin/env python3
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import json 
from flufl.lock import Lock
import pylandau
from datetime import datetime
import lifetime_funcs as LT

from nearline_util import date_from_filename


def main(input_file, output_file_plot, output_file_json):
    segments = []

    print('Opening File ')

    f = h5py.File(input_file)

    segments = f['analysis/rock_muon_segments/data']
    
    mask = (segments['dx'] != 0)

    segments_masked = segments[mask]
    print(f"Extracting Lifetime using {len(segments_masked)} segments")

    timestamp = date_from_filename(input_file)
     
    lifetime, lifetime_error, fig = LT.langau_lifetime(nhits=segments_masked['nhits']/segments_masked['dx'],
                                                  dqdx=segments_masked['dQ']/segments_masked['dx'],
                                                  time_drifted=segments_masked['t'],
                                                  time_bins=np.linspace(0, 1960, 20),
                                                  dqdx_bins=np.linspace(0,250,70),
                                                  nhits_bins=np.linspace(0, 30, 42), 
                                                  wanted_title=timestamp)
    
    print(f"Timestamp: {timestamp}, Lifetime: {lifetime} +- {lifetime_error}")

    with Lock(output_file_json + '.lock'):
        if os.path.exists(output_file_json):
            with open(output_file_json, "r") as f:
                data = json.load(f)
        else:
            data = {"lifetimes": []}

        new_entry = {
            "timestamp": timestamp.isoformat(),
            "lifetime_us": lifetime,
            "error_us": lifetime_error
        }
        data["lifetimes"].append(new_entry)
        with open(output_file_json, "w") as f:
            json.dump(data, f, indent=2)


    fig.savefig(output_file_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, \
                        help="Path to muon hdf5 file", required=True)
     

    parser.add_argument('--output_file_plot', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    
    parser.add_argument('--output_file_json', default=None, \
                        type=str, help='''Output josn file''',
                        required=True)
    
    args = parser.parse_args()
    print(args)
    main(**vars(args))

