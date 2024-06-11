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

def main(input_file, output_file):
    
    pedestal = defaultdict(lambda: dict(pedestal_mv=580))
    
    with open(pedestal_filename, 'r') as infile:
        for key, value in json.load(infile).items():
            pedestal[key] = value
    
    print('Opening file: ', input_file)
    fh = h5py.File(input_file, 'r')
    print('File opened.')
    
    # Calibration

    p = fh['packets']

    clock_tick = 100e-9 # 100ns

    tpcs = range(1, 9)
    
    trigger_tpc = 1
    i_trigger_tpc = trigger_tpc - 1
    
    thresh_tpc = 5
    i_thresh_tpc = thresh_tpc - 1

    sync_ticks = 1. / clock_tick
    sync_plot_range = 1000 # 100 ticks
    sync_bins = np.linspace(-sync_plot_range, sync_plot_range, 2*sync_plot_range)

    drift_len = int(300e-3 / (1.65e-3/1e-6) / clock_tick) #  30cm / (1.65mm/us) / tick
    trigger_bins = np.linspace(-2*drift_len, 4*drift_len, 6*drift_len//10) # 10 tick bins

    # get data from each tpc
    tpc_mask = [p['io_group'] == tpc for tpc in tpcs
               ]
    # find external triggers from A9
    trigger_packet_mask = tpc_mask[i_trigger_tpc] & (p['packet_type'] == 7) & (p['trigger_type'] == 2)

    # find external triggers from threshold
    thresh_packet_mask = tpc_mask[i_thresh_tpc] & (p['packet_type'] == 7) & (p['trigger_type'] == 2)

    # split data at external triggers
    trigger_grps = [
        np.split(p[tpc_mask[i_tpc]], np.argwhere(trigger_packet_mask).flatten())
        for i_tpc in range(len(tpcs))
        ]
    thresh_grps = [
        np.split(p[tpc_mask[i_tpc]], np.argwhere(thresh_packet_mask).flatten())
        for i_tpc in range(len(tpcs))
        ]

    # A9
    
    rel_timestamps = [[
            grp[1:][ (grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)) ]['timestamp'].astype(int) - grp[0]['timestamp'].astype(int)
            for grp in trigger_grps[i_tpc]
            if len(grp) > 1 and np.any((grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)))
            ]
        for i_tpc in range(len(tpcs))
        ]
    
    rel_timestamps = [np.concatenate(ts, axis=0) if len(ts) else np.empty(0,) for ts in rel_timestamps]
    
    rel_q = [[
            charge_from_dataword(grp[1:]['dataword'],
                                 vref_mv,vcm_mv,
                                 np.array([pedestal[unique_id]['pedestal_mv'] 
                                           for unique_id in to_unique_id(grp[1:]).astype(str)
                                          ])
                                )[ (grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)) ]
            for grp in trigger_grps[i_tpc]
            if len(grp) > 1 and np.any((grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)))
            ]
        for i_tpc in range(len(tpcs))
        ]
    
    rel_q = [np.concatenate(qq, axis=0) if len(qq) else np.empty(0,) for qq in rel_q]
        
    trigger_corr_hist = [
        np.histogram(
            np.clip(rel_timestamps[i_tpc], trigger_bins[0], trigger_bins[-1]), bins=trigger_bins, weights=rel_q[i_tpc])[0]
        for i_tpc in range(len(tpcs))
        ]
    
    # Thresh
    
    thresh_rel_timestamps = [[
            grp[1:][ (grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)) ]['timestamp'].astype(int) - grp[0]['timestamp'].astype(int)
            for grp in thresh_grps[i_tpc]
            if len(grp) > 1 and np.any((grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)))
            ]
        for i_tpc in range(len(tpcs))
        ]

    thresh_rel_timestamps = [np.concatenate(ts, axis=0) if len(ts) else np.empty(0,) for ts in thresh_rel_timestamps]

    thresh_rel_q = [[
            charge_from_dataword(grp[1:]['dataword'],
                                 vref_mv,vcm_mv,
                                 np.array([pedestal[unique_id]['pedestal_mv'] 
                                           for unique_id in to_unique_id(grp[1:]).astype(str)
                                          ])
                                )[ (grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)) ]
            for grp in thresh_grps[i_tpc]
            if len(grp) > 1 and np.any((grp[1:]['packet_type']==0) & (grp[1:]['valid_parity'].astype(bool)))
            ]
        for i_tpc in range(len(tpcs))
        ]

    thresh_rel_q = [np.concatenate(qq, axis=0) if len(qq) else np.empty(0,) for qq in thresh_rel_q]

    thresh_corr_hist = [
        np.histogram(
            np.clip(thresh_rel_timestamps[i_tpc], trigger_bins[0], trigger_bins[-1]), bins=trigger_bins, weights=thresh_rel_q[i_tpc])[0]
        for i_tpc in range(len(tpcs))
        ]
    

    
    fig,ax = plt.subplots(2,1,dpi=100,figsize=(10,10))

    for i_tpc in range(len(tpcs)):
        ax[0].plot((trigger_bins[:-1]+trigger_bins[1:])/2, trigger_corr_hist[i_tpc], '.-',
                    alpha=0.5, label='io_group = {}'.format(tpcs[i_tpc]))
        ax[1].plot((trigger_bins[:-1]+trigger_bins[1:])/2, thresh_corr_hist[i_tpc], '.-',
                    alpha=0.5, label='io_group = {}'.format(tpcs[i_tpc]))
        
    ax[0].axvline(0, color='k', linewidth=1, label='A9 trigger')
    ax[0].axvline(drift_len, color='r', linewidth=1, label='Max drift length')
    ax[0].set_xlabel('timestamp relative to external trigger [0.1us]')
    ax[0].set_ylabel('charge [ke-]')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].grid()
    ax[0].autoscale(axis='x', tight=True)
    
    ax[1].axvline(0, color='k', linewidth=1, label='Threshold trigger')
    ax[1].axvline(drift_len, color='r', linewidth=1, label='Max drift length')
    ax[1].set_xlabel('timestamp relative to external trigger [0.1us]')
    ax[1].set_ylabel('charge [ke-]')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].grid()
    ax[1].autoscale(axis='x', tight=True)
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
