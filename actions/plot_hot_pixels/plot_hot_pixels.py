#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import argparse
from datetime import datetime
import numpy as np

from nearline_util import date_from_filename

def main(input_file, output_file):
    
    print('Opening file: ', input_file)
    f = h5py.File(input_file, 'r')
    print('File opened.')

    mask = f['packets']['packet_type'] == 0

    timestamp = date_from_filename(input_file)
    
    print('Counting the number of data packets per channel...')
    key, counts = np.unique(f['packets'][mask][['io_group', 'io_channel', 'chip_id', 'channel_id']], return_counts=True)
    total = np.sum(counts)
    
    print('Sorting...')
    sorting = np.argsort(counts)[-100:]
    print('Sorted.')

    key = key[sorting]
    counts = counts[sorting]

    labels = [f'{int(d["io_group"])} - {d["io_channel"]} - {d["chip_id"]} - {d["channel_id"]}'
              for d in key]
    
    plt.figure(figsize=(7, 16))

    plt.barh(np.arange(len(counts)), counts/np.clip(total,1,None),
            tick_label=labels)
    
    plt.title(timestamp.strftime('%Y/%m/%d %H:%M:%S'))
    
    plt.xlabel('Fraction of data packets')
    plt.ylabel('io_group - io_channel - chip_id - channel')
    plt.xscale('log')
    
    plt.tight_layout()
    
    plt.savefig(output_file)
    print('Saved to: ', output_file)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, \
                        type=str, help='''Path to packet-formatted hdf5 file''', required=True)
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
