#!/usr/bin/env python3

import h5py
import argparse
from datetime import datetime
import numpy as np
import numpy.lib.recfunctions as rfn
import json

import matplotlib.pyplot as plt
from collections import defaultdict

pedestal_filename='packet-cold-pedestal-2024_06_05_08_28_19_CDT_ped.json'

vdda_mv=1800.0
vref_mv=1568.0
vcm_mv =478.1
vref_dac=223
vcm_dac=68

def io_channel_to_tile_id(io_channel):
    return (io_channel - 1)//4 + 1


def charge_from_dataword(dw, vref, vcm, ped):
    return (dw / 256. * (vref - vcm) + vcm - ped) / 4. # hardcoding 4 mV/ke- conv.

def to_unique_id(p):
    return ((p['io_group'].astype(int)*256 \
        + ((p['io_channel'].astype(int)-1)//4+1))*256 \
        + p['chip_id'].astype(int))*64 \
        + p['channel_id'].astype(int)

def date_from_filename(filename):
    date_str = filename.split('-')[-1].split('_CDT')[0]
    timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M_%S")
    return timestamp

def mask_for_data(p, io_group=None):
    m = np.logical_and(p['packet_type'] == 0, p['valid_parity'] == 1)
    if io_group is None:
        return m
    
    return np.logical_and(m, p['io_group'] == io_group)


def main(input_file, output_file):
    
    pedestal = defaultdict(lambda: dict(pedestal_mv=580))
    
    with open(pedestal_filename, 'r') as infile:
        for key, value in json.load(infile).items():
            pedestal[key] = value
            
    beam_start = 2100
    rollover_ticks = 10000000
    clock_tick = 100e-9 # 100ns

    drift_len_nom = int(300e-3 / (1.65e-3/1e-6) / clock_tick) #  30cm / (1.65mm/us) / tick
    drift_len_halfnom = int(300e-3 / (1.07e-3/1e-6) / clock_tick) #  30cm / (1.65mm/us) / tick
    trigger_bins = np.linspace(-0.1*drift_len_nom, 4*drift_len_nom, drift_len_nom//20) # 10 tick bins

    io_groups = range(1, 9)
    trig_io_group = 6

    f = h5py.File(input_file, 'r')
    p = f['packets']

    trigger_mask = np.logical_and(p['packet_type'] == 7, p['trigger_type'] == 2)
    trigger_mask = np.logical_and(trigger_mask, p['io_group'] == trig_io_group)
    trigger_timestamps = p['timestamp'][trigger_mask]

    ts = p['timestamp']

    argtrig = np.argwhere(trigger_mask)
    packets_per_trigger = np.split(p, argtrig.flatten())
    timestamps_per_trigger = np.split(ts, argtrig.flatten())

    rel_timestamps = [[ timestamps_per_trigger[itrig][mask_for_data(packets_per_trigger[itrig], io_group)] - trigger_timestamps[itrig-1] for itrig in range(1, len(argtrig)) ] for io_group in io_groups]
    rel_timestamps = [np.concatenate(rel_timestamps[iog], axis=0) if len(rel_timestamps[iog]) else np.empty(0,) for iog in range(len(io_groups))]
    
    qs = [[ charge_from_dataword(packets_per_trigger[itrig][mask_for_data(packets_per_trigger[itrig], io_group)]['dataword'], vref_mv, vcm_mv,
                                 np.array([pedestal[unique_id]['pedestal_mv'] 
                                           for unique_id in to_unique_id(packets_per_trigger[itrig][mask_for_data(packets_per_trigger[itrig], io_group)]).astype(str)
                                          ])
                                ) for itrig in range(1, len(argtrig)) ] for io_group in io_groups]
    qs = [np.concatenate(qs[iog], axis=0) if len(qs[iog]) else np.empty(0,) for iog in range(len(io_groups))]

    trigger_corr_hist = [np.histogram(
                rel_timestamps[iog], bins=trigger_bins, weights=qs[iog])[0] for iog in range(len(io_groups))]

    fig, ax = plt.subplots(2, 4, figsize=(25,12))

    for iog in range(len(io_groups)):
        ax[iog%2, iog//2].plot((trigger_bins[:-1]+trigger_bins[1:])/2, trigger_corr_hist[iog], 'r-')
        ax[iog%2, iog//2].set_yscale('log')
        ax[iog%2, iog//2].set_xlabel('timestamp relative to external trigger [0.1 us]')
        ax[iog%2, iog//2].set_ylabel('charge [ke-]')
        ax[iog%2, iog//2].set_title(f'io_group = {iog+1}')
        ax[iog%2, iog//2].axvline(0, color='k', linestyle='--', linewidth=1, label='io_group = 6 trig.')
        ax[iog%2, iog//2].axvline(drift_len_halfnom, color='g', linestyle='-', linewidth=1, label='Half-nom. max drift len.')
        ax[iog%2, iog//2].axvline(drift_len_nom, color='b', linestyle='-', linewidth=1, label='Nom. max drift len.')
        if iog == 0:
            ax[iog%2, iog//2].legend()
    
    fig.suptitle(input_file.split('/')[-1])
    plt.draw()

    fig.tight_layout()
    plt.savefig(output_file)
    print('Saved to: ', output_file)
    
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, \
                        type=str, help='''Path to packet-formatted hdf5 file''', required=True)
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    
    args = parser.parse_args()
    main(**vars(args))

    