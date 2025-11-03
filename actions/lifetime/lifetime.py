#!/usr/bin/env python3
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse

import pylandau
from datetime import datetime
import lifetime_funcs as LT

from nearline_util import date_from_filename


def main(input_file, output_file):
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
    fig.savefig(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, \
                        help="Path to muon hdf5 file", required=True)
    
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    
    args = parser.parse_args()
    print(args)
    main(**vars(args))
