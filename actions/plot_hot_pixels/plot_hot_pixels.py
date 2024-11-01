#!/usr/bin/env python3

# Produce sorted tables (by io_group) of pixel data rates (the "top-100")

import matplotlib.pyplot as plt
import h5py
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

def date_from_ped_filename(filename):
    date_str = Path(filename).stem.split('-')[-1]
    # strip the timezone:
    date_str = '_'.join(date_str.split('_')[:-1])
    timestamp = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
    return timestamp


def main(input_file, output_file):

    print('Opening file: ', input_file) # open data file
    f = h5py.File(input_file, 'r')
    print('File opened.')

    unixtime = f['packets'][:]['timestamp'][f['packets'][:]['packet_type'] == 4]
    livetime = np.max(unixtime) - np.min(unixtime)

    mask_data = f['packets']['packet_type'] == 0
    f_data = f['packets'][mask_data][:]

    timestamp = date_from_ped_filename(input_file)

    # the following caches are keyed by io_group
    # and are used to speed up the 4-panel plot
    cached_key, cached_counts = {}, {}

    # make individual top-100 tables for each io_group
    for i in range(1, 5):

        print('IO_group ', i, ': Counting the number of data packets per channel...')

        mask_group = f_data['io_group'] == i

        f_group = f_data[mask_group]

        key, counts = np.unique(f_group[['io_channel', 'chip_id', 'channel_id']], return_counts=True)

        # total = np.sum(counts)
        print('Sorting io_group ', i, ' ...')
        sorting = np.argsort(counts)[-100:]
        print('io_group', i, ' sorted.')

        key = key[sorting]
        counts = counts[sorting]

        cached_key[i] = key
        cached_counts[i] = counts

        labels = [f'{d["io_channel"]} - {d["chip_id"]} - {d["channel_id"]}' for d in key]

        plt.figure(figsize=(7, 16))
        plt.barh(np.arange(len(counts)), counts / livetime, tick_label=labels)
        plt.title(f'IO group {i}, ' + timestamp.strftime('%Y/%m/%d %H:%M:%S'))
        plt.xlabel('Packet Rate (pkts/s)')
        plt.ylabel('io_channel - chip_id - channel')
        plt.xscale('log')
        plt.tight_layout()

        assert output_file.endswith('.png')
        filename = output_file[:-4] + ".io_group_" + str(i) + '.png'
        plt.savefig(filename)
        print("Saved io_group ", i, "to: ", filename)

    # plot a 4-panel version of the top-100 tables
    fig, ax = plt.subplots(1, 4, figsize=(28, 16))
    fig.suptitle('Top-100 Pixel Data Rates, ' + timestamp.strftime('%Y/%m/%d %H:%M:%S'))

    for i in range(1, 5):
        print('Creating the 4-panel version IO_group ', i)

        key = cached_key[i]
        counts = cached_counts[i]

        labels = [f'{d["io_channel"]} - {d["chip_id"]} - {d["channel_id"]}'
                  for d in key]

        ax[i - 1].barh(np.arange(len(counts)), counts / livetime, tick_label=labels)
        ax[i - 1].set_title(f'IO group {i}')
        ax[i - 1].set_xlabel('Packet Rate (pkts/s)')
        ax[i - 1].set_ylabel('io_channel - chip_id - channel')
        ax[i - 1].set_xscale('log')

    plt.tight_layout()
    assert output_file.endswith('.png')
    filename = output_file[:-4] + ".all_io_groups" + '.png'
    plt.savefig(filename)
    print("Saved 4-panel version to: ", filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, \
                        type=str, help='''Path to packet-formatted hdf5 file''', required=True)
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
