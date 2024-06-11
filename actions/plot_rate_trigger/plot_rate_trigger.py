import h5py
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pedestal_filename='/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/global_dac_06_06/reference-cold-pedestal-2024_06_05_08_28_19_CDT.h5'

def date_from_filename(filename):
    date_str = filename.split('-')[-1].split('_CDT')[0]
    timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M_%S")
    return timestamp

def io_channel_to_tile_id(io_channel):
    return (io_channel - 1)//4 + 1

def main(input_file, output_file):
    
    print('Opening file: ', input_file)
    fh = h5py.File(input_file, 'r')
    print('File opened.')

    clock_tick = 100e-9 # 100ns

    tpcs = range(1, 9)
    
    trigger_tpc = 4
    i_trigger_tpc = trigger_tpc - 1
    
    thresh_tpc = 5
    i_thresh_tpc = thresh_tpc - 1

    sync_ticks = 1. / clock_tick
    sync_plot_range = 1000 # 100 ticks
    sync_bins = np.linspace(-sync_plot_range, sync_plot_range, 2*sync_plot_range)

    drift_len = int(300e-3 / (1.65e-3/1e-6) / clock_tick) #  30cm / (1.65mm/us) / tick
    trigger_bins = np.linspace(-2*drift_len, 4*drift_len, 6*drift_len//10) # 10 tick bins

    # get data from each tpc
    tpc_mask = [fh['packets']['io_group'] == tpc for tpc in tpcs
               ]
    # find external triggers from A9
    trigger_packet_mask = tpc_mask[i_trigger_tpc] & (fh['packets']['packet_type'] == 7) & (fh['packets']['trigger_type'] == 2)

    # find external triggers from threshold
    thresh_packet_mask = tpc_mask[i_thresh_tpc] & (fh['packets']['packet_type'] == 7) & (fh['packets']['trigger_type'] == 2)

    # split data at external triggers
    trigger_grps = [
        np.split(fh['packets'][tpc_mask[i_tpc]], np.argwhere(trigger_packet_mask).flatten())
        for i_tpc in range(len(tpcs))
        ]
    thresh_grps = [
        np.split(fh['packets'][tpc_mask[i_tpc]], np.argwhere(thresh_packet_mask).flatten())
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
        
    trigger_corr_hist = [
        np.histogram(
            np.clip(rel_timestamps[i_tpc], trigger_bins[0], trigger_bins[-1]), bins=trigger_bins)[0]
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

    thresh_corr_hist = [
        np.histogram(
            np.clip(thresh_rel_timestamps[i_tpc], trigger_bins[0], trigger_bins[-1]), bins=trigger_bins)[0]
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
    ax[0].set_ylabel('count')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].grid()
    ax[0].autoscale(axis='x', tight=True)
    
    ax[1].axvline(0, color='k', linewidth=1, label='Threshold trigger')
    ax[1].axvline(drift_len, color='r', linewidth=1, label='Max drift length')
    ax[1].set_xlabel('timestamp relative to external trigger [0.1us]')
    ax[1].set_ylabel('count')
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
