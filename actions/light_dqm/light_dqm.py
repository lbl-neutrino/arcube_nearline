#!/usr/bin/env python3

##########################################
##                                      ##
##            ~ Light DQM ~             ##
##                                      ##
##########################################
##
##  Written 09.07.2025
##  Updated 27.08.2025
##
##    - James Mead <jmead@nikhef.nl>
##    - Sindhujha Kumaran <s.kumaran@uci.edu>
##
###########################################
'''
    # example usage:

python light_dqm.py
                       --input_path /global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light/warm_commission/
                       --file_syntax mpd_run_dbg_rctl_
                       --channel_status_file light_dqm/channel_status.csv
                       --output_dir dqm_plots/
                       --tmp_dir tmp/
                       --units ADC16
                       --ptps16bit 150
                       --start_run 0
                       --nfiles 10
                       --ncomp 5
                       --powspec_nevts 200
                       --max_evts 500
                       --write_json_blobs False
                       --merge_grafana_plots False
                       --plot_all_clipped False
                       --plot_all_negatives False
'''
############################################

import numpy as np
import pandas as pd
import h5py
import argparse
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import beta
import json
import sys
import os
import time
import glob
from PyPDF2 import PdfMerger
from matplotlib.lines import Line2D

# install PyPDF2 if not already installed
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("PyPDF2 not found. Please install it using 'pip install PyPDF2'.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Process and plot DUNE light file data.")
    parser.add_argument('--input_path', type=str, default='.', help='Path to input file')
    parser.add_argument('--file_syntax', type=str, default='.', help='File name syntax')
    parser.add_argument('--channel_status_file', type=str, default='actions/light_dqm/channel_status.csv', help='Channel status file')
    parser.add_argument('--output_dir', type=str, default='dqm_plots/', help='Directory to save final output plots')
    parser.add_argument('--tmp_dir', type=str, default='tmp/', help='Directory to save temporary output plots')
    parser.add_argument('--units', type=str, default='ADC16', choices=['ADC16', 'ADC14', 'V'], help='Units for waveform')
    parser.add_argument('--ptps16bit', type=int, default=500, help='Peak-to-peak threshold for 16-bit ADC')
    parser.add_argument('--start_run', type=int, default=0, help='Start run for processing')
    parser.add_argument('--nfiles', type=int, default=1, help='Number of files to process')
    parser.add_argument('--ncomp', type=int, default=-1, help='Number of previous files to compare')
    parser.add_argument('--powspec_nevts', type=int, default=500, help='Number of events to process per file for noise spectra')
    parser.add_argument('--max_evts', type=int, default=500, help='Maximum number of events to process for the whole file')
    parser.add_argument('--write_json_blobs', type=bool, default=False, help='Saves all data to json blobs if true')
    parser.add_argument('--merge_grafana_plots', type=bool, default=False, help='Merge baselines and flatlines grafana plots')
    parser.add_argument('--plot_all_clipped', type=bool, default=False, help='Plot all clipped waveform plots')
    parser.add_argument('--plot_all_negatives', type=bool, default=False, help='Plot all negative baseline plots')
    return parser.parse_args()

# ----------------------------- #
#         Global Constants      #
# ----------------------------- #

args = parse_args()
ptps16bit = args.ptps16bit
default_units = args.units

SAMPLE_RATE = 0.016  # us per sample
SAMPLES = 1000       # samples per waveform
adc14_max = 8191
adc14_16 = 2**2      # Conversion factor between 14 and 16 bit ADC

ADC_V_range = 2.0    # Voltage range of ADC
ADC_V_offset = -1.0  # Voltage offset

# Precomputed peak-to-peak values for different units
ptps_16 = np.array([ptps16bit]*8)
ptps_14 = ptps_16 / adc14_16
ptps_V = ptps_16 * (ADC_V_range) / (adc14_max * adc14_16)

def get_ptps(units):
    """
    Get peak-to-peak values for the specified units.

    Args:
        units (str): 'ADC16', 'ADC14', or 'V'

    Returns:
        np.ndarray: Peak-to-peak values for each ADC
    """
    if units == 'ADC16':
        return ptps_16
    elif units == 'ADC14':
        return ptps_14
    elif units == 'V':
        return ptps_V
    else:
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")

# Frequency regions of interest (MHz) and window width (MHz)
vlist = [0.5e6, 1.8e6, 4.6e6, 7.1e6, 8.5e6, 10e6, 11.5e6, 19e6, 20e6, 25e6, 30e6]
window = 0.4e6

# Channel mapping: select channels 4-15 in each group of 16
channels = []
for group_start in range(0, 64, 16):
    channels.extend(range(group_start + 4, min(group_start + 16, 64)))


# Load channel status (0 = good, nonzero = bad)
channel_status_csv = args.channel_status_file
cs = None  # default if load fails

try:
    cs_df = pd.read_csv(channel_status_csv, header=None)
    cs = cs_df.to_numpy()
    print(f"Channel status loaded successfully from: {channel_status_csv}")
except FileNotFoundError:
    print(f"Channel status file not found, skipping: {channel_status_csv}")
except pd.errors.EmptyDataError:
    print(f"Channel status file is empty: {channel_status_csv}")
except Exception as e:
    print(f"Error loading channel status from {channel_status_csv}: {e}")


# ----------------------------- #
#         Core Functions        #
# ----------------------------- #

# def numpy friendly clopper-pearson function
def clopper_pearson(passed, total, interval=0.68):
    alpha = 1 - interval
    lower = beta.ppf(alpha/2, passed, total - passed + 1)
    upper = beta.ppf(1 - alpha/2, passed + 1, total - passed)
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=1.0)
    frac = np.zeros_like(passed, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = passed / total
        frac = np.nan_to_num(frac, nan=0.0)
    # if total is zero, set frac to 0
    frac = np.where(total == 0, 0.0, frac)
    # if frac is 0 and total is not zero, set lower and upper to 0 and 1 respectively
    mask = (frac == 0) & (total != 0)
    lower = np.where(mask, 0.0, lower)
    upper = np.where(mask, 1.0, upper)
    # Calculate errors
    err_low = 100 * (frac - lower)
    err_up = 100 * (upper - frac)
    pct = 100 * frac

    # Ensure negative error is zero when central value is zero (i.e., no entries)
    mask_no_entries = (total == 0) | (pct == 0)
    err_low[mask_no_entries] = 0.0

    return pct, err_low, err_up


# Convert ADC16 counts to voltage
def adc16_to_voltage(adc_counts, mask=None):
    if mask is not None:
        adc_counts = np.where(mask[..., np.newaxis], adc_counts, 0)
    return adc_counts * (ADC_V_range + ADC_V_offset) / (adc14_max * adc14_16)


# Get waveform information for a given event
def get_waveform_info(waveform, units='ADC16', mask=None, ths=None):
    if mask is not None:
        # mask is expected to be a list/array of event indices
        waveform = waveform[mask]
    # check if the waveform is clipped
    clipped = np.any(waveform >= (adc14_max-1)*adc14_16, axis=-1)
    # define units
    if units == 'ADC14':
        waveform = waveform / adc14_16
    elif units == 'V':
        waveform = adc16_to_voltage(waveform)
    elif units != 'ADC16':
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")
    # take the stdandard deviation of the first 50 samples
    noise = np.std(waveform[:, :, :, :50], axis=3)
    # take the mean of the first 50 sample
    baseline = np.mean(waveform[:, :, :, :50], axis=3)
    # subtract the baseline from the waveform
    max_value = np.max(waveform - baseline[:, :, :, np.newaxis], axis=3)
    # check for negative values below ptps
    negs = np.any(waveform - baseline[:, :, :, np.newaxis] < -ths[np.newaxis, :, np.newaxis, np.newaxis], axis=-1)
    # wvfms of specified events
    wvfms = waveform[:, :, :, :]

    # return the baseline, max value, and clipped status
    return wvfms, noise, baseline, max_value, clipped, negs


### GRAFANA PLOTS ###

# function for checking if channels max range flatlines at any point
def check_flatline(max_values, threshold=0.1):
    # check if the max range is flatlined at any point
    flatlined = np.all(max_values < threshold, axis=0)
    return flatlined

# function to plot an 8x64 grid with red cross for flatlined, green circle for not flatlined
def plot_flatline_mask(flatline_mask, channel_status=None, times=None,
                       output_name='flatline_mask.pdf'):
    n_adcs = flatline_mask.shape[0]
    n_channels = flatline_mask.shape[1]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(-0.5, n_adcs - 0.5)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_adcs))
    ax.set_xlabel('Channel')
    ax.set_ylabel('ADC')
    ax.set_xticklabels(np.arange(n_channels))
    ax.set_yticklabels(np.arange(n_adcs))

    for i in range(n_adcs):
        for j in range(n_channels):
            alpha = 1.0
            # Skip inactive channels
            if channel_status is not None and channel_status[i, j] == -1:
                continue
            if flatline_mask[i, j]:
                # Red cross for flatlined
                ax.plot(j, i, marker='x', color='red', markersize=12, markeredgewidth=2)
            else:
                # Green circle for not flatlined
                ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')
            if channel_status is not None and channel_status[i, j] != 0:
                ax.plot(j, i, marker='.', color='black', markersize=10)
    plt.title("Alive and dead channels", y=1.18, fontsize=14)
    plt.tight_layout()

    # add faded line at x=31.5
    ax.axvline(x=31.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    # add faded line at y = 1.5, 3.5, 5.5, 7.5
    for y in range(1, n_adcs):
        if y % 2 == 0:
            ax.axhline(y=y - 0.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)

    # Custom legend handles
    legend_handles = [
        Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=10, label='Ignore known problem channels'),
        Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Dead (call for help)'),
        Line2D([0], [0], marker='o', color='green', markerfacecolor='none', linestyle='None', markersize=10, label='Still living')
    ]
    # Place legend above the plot, in one lineer', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    # add start and end times to the top left
    if times is not None:
        # Place the time text above the plot, aligned with the legend
        ax.text(0.01, 1.15, f'Start: {times[0]}\nEnd:  {times[1]}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # Make the figure slightly larger to fit the legend
    fig.set_size_inches(16, 4)
    # Increase the top margin to fit the legend
    plt.subplots_adjust(top=0.82)
    output_pdf = f"{args.output_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()



def check_baseline(prev_baseline, current_baseline, units='ADC16', threshold=200):
    """
    Compare previous and current baseline arrays.
    Each baseline is a list of 3 numpy arrays: [central, lower, upper], each shape (8, 64).
    Returns a boolean mask (8, 64) where True indicates a significant change.
    """
    # if units == 'ADC14', change threshold to 250
    if units == 'ADC14':
        threshold = 250

    # Unpack
    curr_c, curr_l, curr_u = current_baseline

    # Compute margin for current baseline (max of lower/upper error)
    margin = np.maximum(np.abs(curr_c - curr_l), np.abs(curr_c - curr_u))  # shape (8, 64)
    # If margin exceeds threshold, flag as True
    mask = margin > threshold

    # If previous baseline is None, return mask as is
    if prev_baseline is None:
        return mask

    # Unpack previous baseline
    prev_c, prev_l, prev_u = prev_baseline

    # Compute difference between current and previous central values
    diff = np.abs(curr_c - prev_c)

    # For points where current > previous
    prev_upper = np.abs(curr_c - prev_u)
    diff_margin_up = np.sqrt(np.square(np.abs(curr_c - curr_l)) + np.square(prev_upper))

    # For points where current < previous
    prev_lower = np.abs(curr_c - prev_l)
    diff_margin_down = np.sqrt(np.square(np.abs(curr_c - curr_u)) + np.square(prev_lower))

    # Where current > previous, use diff_margin_up; where current < previous, use diff_margin_down
    greater = curr_c > prev_c
    diff_margin = np.where(greater, diff_margin_up, diff_margin_down)

    # If baselines are equal, no need to check (mask remains as is)
    # Otherwise, flag if diff > 3*diff_margin
    mask |= (diff > 3 * diff_margin)

    return mask


def plot_baseline_mask(baseline_mask, channel_status=None, times=None,
                       output_name='baseline_mask.pdf'):
    """
    Plot an 8x64 grid with red cross for baseline outliers, green circle for normal.
    Optionally mark known bad channels with a black dot.
    """
    n_adcs = baseline_mask.shape[0]
    n_channels = baseline_mask.shape[1]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(-0.5, n_adcs - 0.5)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_adcs))
    ax.set_xlabel('Channel')
    ax.set_ylabel('ADC')
    ax.set_xticklabels(np.arange(n_channels))
    ax.set_yticklabels(np.arange(n_adcs))

    for i in range(n_adcs):
        for j in range(n_channels):
            # Skip inactive channels
            if channel_status is not None and channel_status[i, j] == -1:
                continue
            if baseline_mask[i, j]:
                # Red cross for baseline outlier
                ax.plot(j, i, marker='x', color='red', markersize=12, markeredgewidth=2)
            else:
                # Green circle for normal
                ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')
            if channel_status is not None and channel_status[i, j] != 0:
                ax.plot(j, i, marker='.', color='black', markersize=10)
    plt.title("Channels baseline status", y=1.18, fontsize=14)
    plt.tight_layout()
    # add faded line at x=31.5
    ax.axvline(x=31.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    # add faded line at y = 1.5, 3.5, 5.5, 7.5
    for y in range(1, n_adcs):
        if y % 2 == 0:
            ax.axhline(y=y - 0.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)

    # Custom legend handles
    legend_handles = [
        Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=10, label='Ignore known problem channels'),
        Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Deviation (call for help)'),
        Line2D([0], [0], marker='o', color='green', markerfacecolor='none', linestyle='None', markersize=10, label='Stable')
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    # add start and end times to the top left
    if times is not None:
        # Place the time text above the plot, aligned with the legend
        ax.text(0.01, 1.15, f'Start: {times[0]}\nEnd:  {times[1]}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # Make the figure slightly larger to fit the legend
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(top=0.82)
    output_pdf = f"{args.output_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()


### DQM PLOTS ###

# Get the sum waveform for a given event and units
def get_sum_waveform(waveform, units='ADC16', mask=None, clip=True):
    # waveform shape: (n_events, n_adcs, n_channels, n_samples)
    if mask is not None:
        waveform = waveform[mask]
    if units == 'ADC14':
        waveform = waveform / adc14_16
    elif units == 'V':
        waveform = adc16_to_voltage(waveform)
    elif units != 'ADC16':
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")
    # sum over samples axis
    sum_waveform = np.sum(waveform, axis=0)
    # clip the waveform if requested to mimic non-baselined oscilloscope limit
    if clip:
        sum_waveform = np.clip(sum_waveform, 0, 8192*4 - 1)
    return sum_waveform


# Plot the sum waveform per EPCB for a single event
def plot_sum_waveform(waveform, units='ADC16', i_evt=0, output_name='sum_waveform.pdf'):
    # waveform shape: (n_events, n_adcs, n_channels, n_samples)
    n_epcbs = int(len(channels) // 6)
    n_adcs = waveform.shape[1]
    nrows = n_adcs * (n_epcbs // 2)
    ncols = 2
    _, axes = plt.subplots(nrows, ncols, figsize=(20, 2 * nrows), sharex=True)

    # Ensure axes is always 2D
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i * (n_epcbs // 2) + (j // 2)
            tt_idx = j % 2
            channels_list = channels[j * 6:(j + 1) * 6]
            # Extract waveform for this event, ADC, and EPCB channels
            epcb_waveform = waveform[i_evt, i, channels_list, :]
            # loop over channels
            for ch in range(len(channels_list)):
                axes[idx, tt_idx].plot(epcb_waveform[ch, :], label=f'Channel {channels_list[ch]}')
            # plot total
            sum_wvfm = get_sum_waveform(epcb_waveform, units=units)
            # set title
            axes[idx, tt_idx].set_title(f'Event {i_evt} - Non-beam trigger timing')
            axes[idx, tt_idx].plot(sum_wvfm, color='black', alpha=0.5, label='Sum ADC')
            axes[idx, tt_idx].set_title(f'ADC {i} - EPCB {j} summed waveforms')
            axes[idx, tt_idx].set_ylabel(f'{units} counts')
            axes[idx, tt_idx].legend(loc='upper right', fontsize='small')
            axes[idx, tt_idx].grid(True)
            axes[idx, tt_idx].set_xlim(0, 1000)
            axes[idx, tt_idx].set_ylim(-5000, 70000)
    axes[-1, 0].set_xlabel('Samples/ time (ticks)')
    axes[-1, 1].set_xlabel('Samples/ time (ticks)')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()


# Plot noise for each ADC channel
def plot_noises(prev_noises, noises, i_evt, mask_inactive=True,
                format_bad_channels=True, output_name='noises.pdf'):
    n_adcs = noises.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]

    medians = np.zeros((n_adcs, noises.shape[2]))
    lowers = np.zeros((n_adcs, noises.shape[2]))
    uppers = np.zeros((n_adcs, noises.shape[2]))

    for i in range(n_adcs):
        ax = axes[i]
        channels_idx = np.arange(noises.shape[2])
        # mask inactive channels
        if mask_inactive:
            mask = np.isin(channels_idx, channels)
        else:
            mask = np.ones_like(channels_idx, dtype=bool)
            # Default values
        median = np.zeros(noises.shape[2])
        lower  = np.zeros(noises.shape[2])
        upper  = np.zeros(noises.shape[2])

        if isinstance(i_evt, np.ndarray) and len(i_evt) > 1:
            # Compute 68% central quantile range (16th and 84th percentiles)
            q16 = np.percentile(noises[i_evt, i, :], 16, axis=0)
            q84 = np.percentile(noises[i_evt, i, :], 84, axis=0)
            median = np.median(noises[i_evt, i, :], axis=0)
            lower = median - q16
            upper = q84 - median
            # Set alpha to 0.5 for bad channels (cs != 0)
            if format_bad_channels and cs.shape == (8, 64):
                alphas = np.ones_like(median)
                bad_mask = cs[i, :] != 0
                alphas[bad_mask] = 0.25

                # Plot previous files with error bars
                if prev_noises is not None:
                    prev_centre, prev_lower, prev_upper = prev_noises
                    prev_centre = prev_centre[i, mask]
                    prev_lower = prev_lower[i, mask]
                    prev_upper = prev_upper[i, mask]
                    for ch, c, l, u in zip(
                        channels_idx[mask], prev_centre, prev_lower, prev_upper
                    ):
                        ax.step(
                            [ch - 0.5, ch + 0.5], [c, c],
                            color='red', alpha=alphas[ch]/2, where='post', linewidth=1
                        )
                        ax.fill_between(
                            [ch - 0.5, ch + 0.5], [c - l, c - l], [c + u, c + u],
                            color='red', alpha=alphas[ch]/4, step='post'
                        )
                # Plot current noise with error bars
                for ch in channels_idx[mask]:
                    ax.errorbar(
                        ch, median[ch], yerr=[[lower[ch]], [upper[ch]]],
                        fmt='.', color='b', markersize=3,
                        linewidth=0.5, capsize=5, capthick=1,
                        alpha=alphas[ch]
                    )
        # formatting
        ax.set_title(f'ADC {i} Noise')
        ax.set_ylabel('Noise (ADC counts)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Store median, lower, and upper for later use
        medians[i, mask] = median[mask]
        lowers[i, mask] = lower[mask]
        uppers[i, mask] = upper[mask]
    axes[-1].set_xlabel('channel #')
    # adjust layout
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return medians, lowers, uppers


# Plot baselines for each ADC channel
def plot_baselines(prev_baselines, baselines, i_evt, mask_inactive=True,
                   format_bad_channels=True, output_name='baselines.pdf'):
    n_adcs = baselines.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]
    # Initialize arrays to store median, lower, and upper bounds
    medians = np.median(baselines[i_evt,:,:], axis=0)
    lowers = np.percentile(baselines[i_evt,:,:], 16, axis=0)
    uppers = np.percentile(baselines[i_evt,:,:], 84, axis=0)
    # Loop over each ADC
    for i in range(n_adcs):
        ax = axes[i]
        channels_idx = np.arange(baselines.shape[2])
        # mask inactive channels
        if mask_inactive:
            mask = np.isin(channels_idx, channels)
        else:
            mask = np.ones_like(channels_idx, dtype=bool)
        
        if isinstance(i_evt, np.ndarray) and len(i_evt) > 1:
            # Compute 68% central quantile range (16th and 84th percentiles)
            q16 = lowers[i, mask]
            q84 = uppers[i, mask]
            median = medians[i, mask]
            lower = median - q16
            upper = q84 - median

            # Set alpha to 0.5 for bad channels (cs != 0)
            if format_bad_channels and cs.shape == (8, 64):
                alphas = np.ones_like(median)
                bad_mask = cs[i, :] != 0
                alphas[bad_mask] = 0.25

                # Plot previous files with error bars
                if prev_baselines is not None:
                    prev_centre, prev_lower, prev_upper = prev_baselines
                    prev_centre = prev_centre[i, mask]
                    prev_lower = prev_lower[i, mask]
                    prev_upper = prev_upper[i, mask]
                    for ch, c, l, u in zip(
                        channels_idx[mask], prev_centre, prev_lower, prev_upper
                    ):
                        ax.step(
                            [ch - 0.5, ch + 0.5], [c, c],
                            color='red', alpha=alphas[ch]/2, where='post', linewidth=1
                        )
                        ax.fill_between(
                            [ch - 0.5, ch + 0.5], [c - l, c - l], [c + u, c + u],
                            color='red', alpha=alphas[ch]/4, step='post'
                        )
                # Plot current baseline with error bars
                for ch in channels_idx[mask]:
                    ax.errorbar(
                        ch, median[ch], yerr=[[lower[ch]], [upper[ch]]],
                        fmt='.', color='b', markersize=3,
                        linewidth=0.5, capsize=5, capthick=1,
                        alpha=alphas[ch]
                    )
        # formatting
        ax.set_title(f'ADC {i} Baselines')
        ax.set_ylabel('Baseline (ADC counts)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # adjust layout
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return medians, lowers, uppers


def get_max_value_mask(max_values, ptps, cs=None):
    # if ptps is a single value, convert it to a list of the same length as the number of ADCs
    if isinstance(ptps, (int, float)):
        ptps = [ptps] * 8
    elif len(ptps) != 8:
        raise ValueError("ptps must be a single value or a list of length 8")
    # Vectorized mask: True where max_values < ptps[adc] and channel_status == 0
    max_mask = max_values < (np.array(ptps)[np.newaxis, :, np.newaxis])
    if cs is not None:
        ch_mask = (cs == 0)[np.newaxis, :, :]
        mask = max_mask & ch_mask
    else:
        mask = max_mask
    return mask

def get_noise_spectra(waveform, mask=None):
    # mask waveform, set to Nan
    if mask is not None:
        waveform = np.where(mask[..., np.newaxis], waveform, np.nan)
    # calculate the FFT of the waveform
    fft_N = rfft(waveform, axis=-1) / int(SAMPLES/2+1)
    # calculate the power spectrum
    power_spectra = 2*np.abs(fft_N)**2
    upper_quantile, power_spectrum, lower_quantile = np.nanquantile(power_spectra, [0.84, 0.5, 0.16], axis=0)
    # calculate the frequency bins
    freq_bins = rfftfreq(SAMPLES, d=SAMPLE_RATE * 1e-6)
    return freq_bins, power_spectra, power_spectrum, upper_quantile, lower_quantile

# plot the average noise spectrum
def plot_noise_spectra_epcb(
    freq_bins, noise_spectra, upper_quantile, lower_quantile,
    skip_bad_channels=True, nevts=None, output_name='noise_spectra_epcb.pdf'):

    n_epcbs = int(len(channels)/6)
    n_adcs = noise_spectra.shape[0]
    axes = plt.subplots(n_adcs, 1, figsize=(20, 2*n_adcs), sharex=True)[1]

    # Mask the first frequency bin (set to nan)
    freq_mask = np.ones_like(freq_bins, dtype=bool)
    freq_mask[0] = False

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i
             # mask for epcb channels
            epcb_mask = np.zeros(noise_spectra.shape[1], dtype=bool)
            channels_list = channels[j*6:(j+1)*6]
            epcb_mask[channels_list] = True
            # Mask for good channels
            if skip_bad_channels and cs.shape == (8, 64):
                ch_mask = (cs[i, :] == 0)
            else:
                ch_mask = np.ones(noise_spectra.shape[1], dtype=bool)
            ch_mask = np.logical_and(ch_mask, epcb_mask)
 
            # Average over channels
            avg = np.nanmean(noise_spectra[i, ch_mask], axis=0)
            axes[idx].step(
                freq_bins[freq_mask]/1e6,
                avg[freq_mask],
                where='mid',
                label=f'EPCB {j}'
            )

            axes[idx].set_title(f'ADC {i}, Average Noise Spectrum per EPCB: equidistant {nevts} events')
            axes[idx].set_ylabel('V^2 / bin')
            axes[idx].set_yscale('log')
            axes[idx].set_ylim(5e-10, 1e-5)
            axes[idx].set_xlim(0, 31.25)
            axes[idx].legend(loc='upper right', fontsize='x-small')

            # add grid lines
            axes[idx].grid(True, which='major', linestyle='-', linewidth=0.5)

    axes[-1].set_xlabel('Frequency (MHz)')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

# plot the channel by channel noise spectrum
def plot_noise_spectra_channels(
    freq_bins, noise_spectra, upper_quantile, lower_quantile,
    skip_bad_channels=True, nevts=None, output_name='noise_spectra_channels.pdf'):

    n_epcbs = int(len(channels)/6)
    n_adcs = noise_spectra.shape[0]
    axes = plt.subplots(n_adcs*n_epcbs, 1, figsize=(20, 2*n_adcs*n_epcbs), sharex=True)[1]

    # Mask the first frequency bin (set to nan)
    freq_mask = np.ones_like(freq_bins, dtype=bool)
    freq_mask[0] = False

    # defining regions of interest
    vlist = [0.5e6, 1.8e6, 4.6e6, 7.1e6, 8.5e6, 10e6, 11.5e6, 19e6, 20e6, 25e6, 30e6]
    window = 0.4e6

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i * n_epcbs + j
             # mask for epcb channels
            epcb_mask = np.zeros(noise_spectra.shape[1], dtype=bool)
            channels_list = channels[j*6:(j+1)*6]
            epcb_mask[channels_list] = True
            # Mask for good channels
            if skip_bad_channels and cs.shape == (8, 64):
                ch_mask = (cs[i, :] == 0)
            else:
                ch_mask = np.ones(noise_spectra.shape[1], dtype=bool)
            ch_mask = np.logical_and(ch_mask, epcb_mask)
   
            # Plot each epcb
            for k in ch_mask.nonzero()[0]:

                # plot step for centre line and fill between for error bands
                axes[idx].step(
                    freq_bins[freq_mask]/1e6,
                    noise_spectra[i, k][freq_mask],
                    where='mid', label=f'Channel {k}'
                )

            # Add horizontal line at the minimum value
            min_val = np.nanmin(noise_spectra[i, :, freq_mask])
            axes[idx].axhline(y=min_val, color='k', linestyle='--')
            axes[idx].set_title(f'ADC {i} EPCB {j} average noise spectrum: equidistant {nevts} events')
            axes[idx].legend(loc='upper right', fontsize='small')

            axes[idx].set_ylabel('V^2 / bin')
            axes[idx].set_yscale('log')
            axes[idx].grid(True, which='major', linestyle='-', linewidth=0.5)
            axes[idx].set_ylim(5e-10, 1e-5)
            axes[idx].set_xlim(0, 31.25)

    axes[-1].set_xlabel('Frequency (MHz)')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()


# plot clipped as a histogram with optional Clopper-Pearson error bars
def plot_clipped_fraction(prev_clipped_evts, clipped_evts, title=None,
                          output_name='clipped_fraction.pdf'):
    n_evts = clipped_evts.shape[0]
    n_adcs = clipped_evts.shape[1]
    n_chs = clipped_evts.shape[2]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]
    # initialize arrays to store numerator and denominator for each ADC
    clip_passed = np.zeros((n_adcs, n_chs))
    total_evts = np.zeros((n_adcs, n_chs))

    # Define channels to plot
    for i_adc in range(n_adcs):
        ax = axes[i_adc]

        # Calculate clipped counts for this ADC
        clipped_counts = np.sum(clipped_evts[:,i_adc,:], axis=0)
        # get total events for this file
        total_events = np.array([n_evts] * n_chs) # total events per channel

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            clipped_pct = 100 * clipped_counts / total_events
            clipped_pct = np.nan_to_num(clipped_pct, nan=0.0)

        # Clopper-Pearson interval (binomial proportion confidence interval)
        clipped_pct, err_low, err_up = clopper_pearson(clipped_counts, total_events)
        ylabel = 'Clipped (% total)'

        # define alpha 0.25 for cs!=0
        alphas = np.ones_like(clipped_pct)
        if cs.shape == (8, 64):
            bad_mask = cs[i_adc, :] != 0
            alphas[bad_mask] = 0.25
        # Plot each point individually to allow per-point alpha
        for idx in channels:
            # previous clipped events
            if prev_clipped_evts is not None:
                prev_centre, prev_lower, prev_upper = prev_clipped_evts
                c = prev_centre[i_adc, idx]
                l = prev_lower[i_adc, idx]
                u = prev_upper[i_adc, idx]
                ax.step(
                    [idx - 0.5, idx + 0.5], [c, c],
                    color='red', alpha=alphas[idx]/2, where='post', linewidth=1
                )
                ax.fill_between(
                    [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                    color='red', alpha=alphas[idx]/4, step='post'
                )
            color = 'b'
            if clipped_pct[idx] == 0:  
                # just a single point at zero
                ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
            else:
                # point + Clopper–Pearson error bars
                yerr_lower = max(err_low[idx], 0)
                yerr_upper = max(err_up[idx], 0)
                ax.errorbar(
                    idx, clipped_pct[idx],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='.', ecolor=color, markersize=3,
                    capsize=4, linewidth=1, color=color,
                    alpha=alphas[idx]
                )


        # formatting
        #ax.set_ylim(0, 1)  # Set y-axis limit to 0-10%
        ax.set_title(f'ADC {i_adc} - '+ title if title else 'ADC {i_adc}')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # save clipped numerator and denominator for later use
        clip_passed[i_adc, :] = clipped_counts
        total_evts[i_adc, :] = total_events
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return clip_passed, total_events


# plot clipped as a histogram with optional Clopper-Pearson error bars
def plot_clipped_tpc_fraction(prev_clipped_evts, clipped_evts, max_vals, ths,
                              title=None, output_name='clipped_tpc_fraction.pdf'):
    n_adcs = clipped_evts.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]
    # initialize arrays to store numerator and denominator for each ADC
    clip_passed = np.zeros((n_adcs, 64))
    total_evts = np.zeros((n_adcs, 64))

    # loop over each ADC
    for i_adc in range(n_adcs):
        ax = axes[i_adc]

        # get total events for this file: for each TPC, count events where any channel in the TPC is over ptps
        total_events = np.zeros((64), dtype=int)

        # Calculate clipped counts for this ADC
        clipped_counts = np.sum(clipped_evts[:, i_adc, :], axis=0)
        dir = 1 if i_adc % 2 == 0 else -1
        tpc_range = [i_adc, i_adc + dir]

        # Loop over each TPC
        for i_tpc in tpc_range:
            # Each TPC is associated with two ADCs and 32 channels
            dir = 1 if i_tpc % 2 == 0 else -1
            adc_idx = [i_tpc, i_tpc + dir]
            ch_idx = range(0, 32) if i_tpc % 2 == 0 else range(32, 64)
            # Select the relevant max_vals for these ADCs and channels
            mv = max_vals[:, adc_idx, :][:, :, ch_idx]  # shape: (n_events, 2, 32)
            ptps_sel = ths[adc_idx][:, np.newaxis]  # shape: (2, 1)
            # Find events where any channel in the TPC is over ptps
            over_ptps = np.any(mv > ptps_sel, axis=(1, 2))  # shape: (n_events,)
            count = np.sum(over_ptps)
            for ch in ch_idx:
                total_events[ch] = count

        # Clopper-Pearson interval (binomial proportion confidence interval)
        clipped_pct, err_low, err_up = clopper_pearson(clipped_counts, total_events)
        ylabel = 'Clipped (% TPC)'

        # define alpha 0.25 for cs!=0
        alphas = np.ones_like(cs[i_adc, :])
        bad_mask = cs[i_adc, :] != 0
        alphas[bad_mask] = 0.25
        # Plot each point individually to allow per-point alpha
        for idx in channels:
            # previous clipped events
            if prev_clipped_evts is not None:
                prev_centre, prev_lower, prev_upper = prev_clipped_evts
                c = prev_centre[i_adc, idx]
                l = prev_lower[i_adc, idx]
                u = prev_upper[i_adc, idx]
                ax.step(
                    [idx - 0.5, idx + 0.5], [c, c],
                    color='red', alpha=alphas[idx]/2, where='post', linewidth=1
                )
                ax.fill_between(
                    [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                    color='red', alpha=alphas[idx]/4, step='post'
                )
            color = 'b'
            if clipped_pct[idx] == 0:  
                # just a single point at zero
                ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
            else:
                # point + Clopper–Pearson error bars
                yerr_lower = max(err_low[idx], 0)
                yerr_upper = max(err_up[idx], 0)
                ax.errorbar(
                    idx, clipped_pct[idx],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='.', ecolor=color, markersize=3,
                    capsize=4, linewidth=1, color=color,
                    alpha=alphas[idx]
                )


        # formatting
        #ax.set_ylim(0, 5)  # Set y-axis limit to 0-10%
        ax.set_title(f'ADC {i_adc} - {title}' if title else f'ADC {i_adc}')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # save clipped numerator and denominator for later use
        clip_passed[i_adc, :] = clipped_counts
        total_evts[i_adc, :] = total_events
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return clip_passed, total_evts


# plot clipped as a histogram with optional Clopper-Pearson error bars
def plot_clipped_epcb_fraction(prev_clipped_evts, clipped_evts, max_vals, ths,
                               title=None, output_name='clipped_epcb_fraction.pdf'):
    n_adcs = clipped_evts.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]
    # Initialize arrays to store numerator and denominator for each ADC
    clip_passed = np.zeros((n_adcs, 64))
    total_evts = np.zeros((n_adcs, 64))

    # Calculate max_values for each ADC
    for i_adc in range(n_adcs):
        ax = axes[i_adc]

        # Calculate clipped counts for this ADC
        clipped_counts = np.sum(clipped_evts[:,i_adc,:], axis=0)

        # get total events for this adc with max_values > ptps
        total_events = np.zeros((64))
        for i_epcb in range(8):
            channel_indices = channels[i_epcb * 6: (i_epcb + 1) * 6]
            channel_mask = np.zeros(64, dtype=bool)
            channel_mask[channel_indices] = True
            over_ptps = np.any(max_vals[:, i_adc, :]*channel_mask > ths[np.newaxis, i_adc, np.newaxis], axis=-1)
            total_events[channel_indices] = np.sum(over_ptps, axis=0)

        # Clopper-Pearson interval (binomial proportion confidence interval)
        clipped_pct, err_low, err_up = clopper_pearson(clipped_counts, total_events)
        ylabel = 'Clipped (% EPCB)'

        # define alpha 0.25 for cs!=0
        alphas = np.ones_like(cs[i_adc, :])
        bad_mask = cs[i_adc, :] != 0
        alphas[bad_mask] = 0.25
        # Plot each point individually to allow per-point alpha
        for idx in channels:
            # previous clipped events
            if prev_clipped_evts is not None:
                prev_centre, prev_lower, prev_upper = prev_clipped_evts
                c = prev_centre[i_adc, idx]
                l = prev_lower[i_adc, idx]
                u = prev_upper[i_adc, idx]
                ax.step(
                    [idx - 0.5, idx + 0.5], [c, c],
                    color='red', alpha=alphas[idx]/2, where='post', linewidth=1
                )
                ax.fill_between(
                    [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                    color='red', alpha=alphas[idx]/4, step='post'
                )

            color = 'b'
            if clipped_pct[idx] == 0:  
                # just a single point at zero
                ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
            else:
                # point + Clopper–Pearson error bars
                yerr_lower = max(err_low[idx], 0)
                yerr_upper = max(err_up[idx], 0)
                ax.errorbar(
                    idx, clipped_pct[idx],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='.', ecolor=color, markersize=3,
                    capsize=4, linewidth=1, color=color,
                    alpha=alphas[idx]
                )

        # formatting
        #ax.set_ylim(0, 5)  # Set y-axis limit to 0-10%
        ax.set_title(f'ADC {i_adc}' + (f' - {title}' if title else 'ADC {i_adc}'))
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # save clipped numerator and denominator for later use
        clip_passed[i_adc, :] = clipped_counts
        total_evts[i_adc, :] = total_events
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return clip_passed, total_evts


# plot clipped as a histogram with optional Clopper-Pearson error bars
def plot_clipped_ch_fraction(prev_clipped_evts, clipped_evts, max_vals, ths,
                             title=None, output_name='clipped_ch_fraction.pdf'):
    n_adcs = clipped_evts.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]

    # Initialize arrays to store numerator and denominator for each ADC
    clip_passed = np.zeros((n_adcs, 64))
    total_evts = np.zeros((n_adcs, 64))

    # Calculate max_values for each ADC
    for i_adc in range(n_adcs):
        ax = axes[i_adc]

        # Calculate clipped counts for this ADC
        clipped_counts = np.sum(clipped_evts[:,i_adc,:], axis=0)
        # get total events for this adc with max_values > ptps
        total_events = np.sum(max_vals[:, i_adc, :] > ths[np.newaxis, i_adc, np.newaxis], axis=0)

        # Clopper-Pearson interval (binomial proportion confidence interval)
        clipped_pct, err_low, err_up = clopper_pearson(clipped_counts, total_events)
        ylabel = 'Clipped (%)'

        # define alpha 0.25 for cs!=0
        alphas = np.ones_like(cs[i_adc, :])
        bad_mask = cs[i_adc, :] != 0
        alphas[bad_mask] = 0.25
        # Plot each point individually to allow per-point alpha
        for idx in channels:
            # previous clipped events
            if prev_clipped_evts is not None:
                prev_centre, prev_lower, prev_upper = prev_clipped_evts
                c = prev_centre[i_adc, idx]
                l = prev_lower[i_adc, idx]
                u = prev_upper[i_adc, idx]
                ax.step(
                    [idx - 0.5, idx + 0.5], [c, c],
                    color='red', alpha=alphas[idx]/2, where='post', linewidth=1
                )
                ax.fill_between(
                    [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                    color='red', alpha=alphas[idx]/4, step='post'
                )
            color = 'b'
            if clipped_pct[idx] == 0:  
                # just a single point at zero
                ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
            else:
                # point + Clopper–Pearson error bars
                yerr_lower = max(err_low[idx], 0)
                yerr_upper = max(err_up[idx], 0)
                ax.errorbar(
                    idx, clipped_pct[idx],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='.', ecolor=color, markersize=3,
                    capsize=4, linewidth=1, color=color,
                    alpha=alphas[idx]
                )

        # formatting
        #ax.set_ylim(0, 10)  # Set y-axis limit to 0-10%
        ax.set_title(f'ADC {i_adc} - ' + (title if title else f'ADC {i_adc}'))
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # save numerator and denominator for later use
        clip_passed[i_adc, :] = clipped_counts
        total_evts[i_adc, :] = total_events
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return clip_passed, total_evts


# plot negative spikes as a histogram with optional Clopper-Pearson error bars
def plot_neg_tpc_fraction(prev_neg_evts, neg_evts, max_vals, ths,
                          title=None, output_name='neg_tpc_fraction.pdf'):
  n_adcs = neg_evts.shape[1]
  axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]

  # Initialize arrays to store numerator and denominator for each ADC
  neg_passed = np.zeros((n_adcs, 64))
  total_evts = np.zeros((n_adcs, 64))

  for i_adc in range(n_adcs):
    ax = axes[i_adc]

    # get total events for this file: for each TPC, count events where any channel in the TPC is over ptps
    total_events = np.zeros((64), dtype=int)

    # Calculate negative spike counts for this ADC
    neg_counts = np.sum(neg_evts[:, i_adc, :], axis=0)
    dir = 1 if i_adc % 2 == 0 else -1
    tpc_range = [i_adc, i_adc + dir]

    for i_tpc in tpc_range:
      dir = 1 if i_tpc % 2 == 0 else -1
      adc_idx = [i_tpc, i_tpc + dir]
      ch_idx = range(0, 32) if i_tpc % 2 == 0 else range(32, 64)
      mv = max_vals[:, adc_idx, :][:, :, ch_idx]
      ptps_sel = ths[adc_idx][:, np.newaxis]
      over_ptps = np.any(mv > ptps_sel, axis=(1, 2))
      count = np.sum(over_ptps)
      for ch in ch_idx:
        total_events[ch] = count

    # Clopper-Pearson interval (binomial proportion confidence interval)
    neg_pct, err_low, err_up = clopper_pearson(neg_counts, total_events)
    ylabel = '-ve baseline (% TPC)'

    # define alpha 0.25 for cs!=0
    alphas = np.ones_like(cs[i_adc, :])
    bad_mask = cs[i_adc, :] != 0
    alphas[bad_mask] = 0.25
    # Plot each point individually to allow per-point alpha
    for idx in channels:
        # previous negative events
        if prev_neg_evts is not None:
            prev_centre, prev_lower, prev_upper = prev_neg_evts
            c = prev_centre[i_adc, idx]
            l = prev_lower[i_adc, idx]
            u = prev_upper[i_adc, idx]
            ax.step(
                [idx - 0.5, idx + 0.5], [c, c],
                color='red', alpha=alphas[idx]/2, where='post', linewidth=1
            )
            ax.fill_between(
                [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                color='red', alpha=alphas[idx]/4, step='post'
            )
        # current file
        color = 'b'
        if neg_pct[idx] == 0:  
            # just a single point at zero
            ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
        else:
            # point + Clopper–Pearson error bars
            yerr_lower = max(err_low[idx], 0)
            yerr_upper = max(err_up[idx], 0)
            ax.errorbar(
                idx, neg_pct[idx],
                yerr=[[yerr_lower], [yerr_upper]],
                fmt='.', ecolor=color, markersize=3,
                capsize=4, linewidth=1, color=color,
                alpha=alphas[idx]
            )


    # formatting
    #x.set_ylim(0,100)
    ax.set_title(f'ADC {i_adc} - ' + (title if title else f'ADC {i_adc} negative baselines'))
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # save numerator and denominator for later use
    neg_passed[i_adc, :] = neg_counts
    total_evts[i_adc, :] = total_events
    axes[-1].set_xlabel('channel #')
  plt.tight_layout()
  plt.show()
  # save as pdf
  output_pdf = f"{args.tmp_dir}/{output_name}"
  with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
  return neg_passed, total_evts


# plot negative spikes as a histogram with optional Clopper-Pearson error bars
def plot_neg_epcb_fraction(prev_neg_evts, neg_evts, max_vals, ths,
                           title=None, output_name='neg_epcb_fraction.pdf'):
  n_adcs = neg_evts.shape[1]
  axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]

  # Initialize arrays to store numerator and denominator for each ADC
  neg_passed = np.zeros((n_adcs, 64))
  total_evts = np.zeros((n_adcs, 64))

  for i_adc in range(n_adcs):
    ax = axes[i_adc]

    # Calculate negative spike counts for this ADC
    neg_counts = np.sum(neg_evts[:, i_adc, :], axis=0)

    # get total events for this adc with max_values > ptps
    total_events = np.zeros((64))
    for i_epcb in range(8):
      channel_indices = channels[i_epcb * 6: (i_epcb + 1) * 6]
      channel_mask = np.zeros(64, dtype=bool)
      channel_mask[channel_indices] = True
      over_ptps = np.any(max_vals[:, i_adc, :]*channel_mask > ths[np.newaxis, i_adc, np.newaxis], axis=-1)
      total_events[channel_indices] = np.sum(over_ptps, axis=0)

    # Clopper-Pearson interval (binomial proportion confidence interval)
    neg_pct, err_low, err_up = clopper_pearson(neg_counts, total_events)
    ylabel = '-ve baseline (% EPCB)'

    # define alpha 0.25 for cs!=0
    alphas = np.ones_like(cs[i_adc, :])
    bad_mask = cs[i_adc, :] != 0
    alphas[bad_mask] = 0.25
    # Plot each point individually to allow per-point alpha
    for idx in channels:
        # previous negative events
        if prev_neg_evts is not None:
            prev_centre, prev_lower, prev_upper = prev_neg_evts
            c = prev_centre[i_adc, idx]
            l = prev_lower[i_adc, idx]
            u = prev_upper[i_adc, idx]
            ax.step(
                [idx - 0.5, idx + 0.5], [c, c],
                color='red', alpha=alphas[idx]/2, where='post', linewidth=1
            )
            ax.fill_between(
                [idx - 0.5, idx + 0.5], [c - l, c - l], [c + u, c + u],
                color='red', alpha=alphas[idx]/4, step='post'
            )
        # current file
        color = 'b'
        if neg_pct[idx] == 0:  
            # just a single point at zero
            ax.plot(idx, 0, '.', color=color, markersize=3, alpha=alphas[idx])
        else:
            # point + Clopper–Pearson error bars
            yerr_lower = max(err_low[idx], 0)
            yerr_upper = max(err_up[idx], 0)
            ax.errorbar(
                idx, neg_pct[idx],
                yerr=[[yerr_lower], [yerr_upper]],
                fmt='.', ecolor=color, markersize=3,
                capsize=4, linewidth=1, color=color,
                alpha=alphas[idx]
            )


    # formatting
    ax.set_title(f'ADC {i_adc} - ' + (title if title else f'ADC {i_adc} negative baselines'))
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # save negative percentages and bounds
    neg_passed[i_adc, :] = neg_counts
    total_evts[i_adc, :] = total_events
  axes[-1].set_xlabel('channel #')
  plt.tight_layout()
  plt.show()
  # save as pdf
  output_pdf = f"{args.tmp_dir}/{output_name}"
  with PdfPages(output_pdf) as pdf:
      pdf.savefig()
      plt.close()
  return neg_passed, total_evts

# save data as JSON
def save_as_json(file_index, data_c, data_l, data_u, output_dir, filename):
    """
    Save data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    If the file exists, append to it; otherwise, create a new file.
    """
    data = {
        'file_index': int(file_index),
        'data_c': data_c.tolist(),
        'data_l': data_l.tolist(),
        'data_u': data_u.tolist()
    }
    file_path = output_dir + filename
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as f:
        json.dump(data, f)
        f.write('\n')
def read_from_json(file_indices, output_dir, filename):
    """
    Read data from JSON file, loop over file indices, add in quadrature.
    If the file is missing, unreadable, or no matching entries are found, return None.
    """
    filepath = os.path.join(output_dir, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"JSON file not found: {filepath}, not making comparisons")
        return None

    data_c, data_l, data_u = [], [], []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('file_index') in file_indices:
                        data_c.append(entry['data_c'])
                        data_l.append(entry['data_l'])
                        data_u.append(entry['data_u'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filepath}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if not data_c:
        return None

    data_c = np.array(data_c)
    data_l = np.array(data_l)
    data_u = np.array(data_u)

    # Combine results: mean for central, quadrature sum for uncertainties
    data_c = np.mean(data_c, axis=0)
    data_l = np.sqrt(np.sum(data_l**2, axis=0))
    data_u = np.sqrt(np.sum(data_u**2, axis=0))

    return data_c, data_l, data_u

def save_spectra_as_json(file_index, spectra_roi, output_dir, filename):
    """
    Save frequency spectra data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    """
    data = {
        'file_index': int(file_index),
        'spectra_roi': spectra_roi.tolist()
    }
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(output_dir + filename) else 'w'
    with open(output_dir+filename, mode) as f:
        json.dump(data, f)
        f.write('\n')

def save_eff_as_json(file_index, passed, totals, output_dir, filename):
    """
    Save efficiency data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    """
    data = {
        'file_index': int(file_index),
        'pass': passed.tolist(),
        'totals': totals.tolist()
    }
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(output_dir + filename) else 'w'
    with open(output_dir+filename, mode) as f:
        json.dump(data, f)
        f.write('\n')
        f.close()

def read_eff_from_json(file_indices, output_dir, filename):
    """
    Read efficiency data from JSON file, loop over file indices,
    combine the passes and totals.
    If the file is missing, unreadable, or no matching entries are found,
    return None.
    """
    filepath = os.path.join(output_dir, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Efficiency JSON file not found: {filepath}, not making comparisons")
        return None

    passed, totals = [], []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('file_index') in file_indices:
                        passed.append(entry['pass'])
                        totals.append(entry['totals'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filepath}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if not passed or not totals:
        return None

    # Sum across all matching entries
    passed = np.sum(np.array(passed), axis=0)
    totals = np.sum(np.array(totals), axis=0)

    return passed, totals

# ----------------------------- #
#             Main              #
# ----------------------------- #

def main():
    """
    Main entry point for DUNE light file processing and plotting.
    """
    # start timer
    start_time = time.time()
    from ascii import header, footer
    header()

    args = parse_args()
    if args.units not in ['ADC16', 'ADC14', 'V']:
        print(f"Invalid units: {args.units}. Must be 'ADC16', 'ADC14', or 'V'.")
        sys.exit(1)
    

    if args.powspec_nevts < args.max_evts:
        raise ValueError("Number of events for the power spectrum can't be lower than the total events")
    # Ensure output directory exists

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory is ready: {args.output_dir}")
    except Exception as e:
        print(f"Error creating output directory {args.output_dir}: {e}")
        raise
    
    try:
        os.makedirs(args.tmp_dir, exist_ok=True)
        print(f"Temporary directory is ready: {args.output_dir}")
    except Exception as e:
        print(f"Error creating temporary directory {args.output_dir}: {e}")
        raise
    # files
    files_arr = np.arange(args.start_run, args.start_run +args.nfiles, 1)
    file_time = start_time
    proc_files = 0
    for i_file in files_arr:

        # Construct the filename based on the input path and file index
        filename = f'{args.input_path}{args.file_syntax}{i_file}.FLOW.hdf5'

        # Skip if file does not exist
        if not os.path.exists(filename):
            print(f"File not found, skipping: {filename}")
            continue
        proc_files += 1
        print(f"Processing file: {filename} with units: {args.units}")

        ptps = get_ptps(args.units)
    
        try:
            file = h5py.File(filename, 'r')
            print(f"File opened successfully: {filename}")
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue

        # Get start and end timestamps in ms
        start_timestamp = file["light/events/data"][0]['utime_ms'][0]
        end_timestamp = file["light/events/data"][-1]['utime_ms'][0]

        # Convert to UTC datetime string
        start_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_timestamp / 1000))
        end_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_timestamp / 1000))
        #print(f"File start time (UTC): {start_utc}")
        #print(f"File end time (UTC): {end_utc}")
        # Convert to US Central Time (UTC-6)
        start_central = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_timestamp / 1000 - 6 * 3600))
        end_central = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_timestamp / 1000 - 6 * 3600))
        print(f"File start time (US Central, UTC-6): {start_central}")
        print(f"File end time (US Central, UTC-6): {end_central}")

        # files for comparison
        if args.ncomp == -1 or args.ncomp > i_file:
            ncomps = np.arange(0, i_file, 1)
        else:
            ncomps = np.arange(i_file-args.ncomp, i_file, 1)
        # total number of available events

        
        nevents_total = file["light/wvfm/data"]['samples'].shape[0]
        if args.max_evts > nevents_total:
            raise ValueError(f"args.max_evts: Total number of events to process {args.max_evts} is less than the number of events in the file {nevents_total}")
        # select evenly spaced indices up to max_evts
        sel_idx = np.linspace(0, nevents_total - 1, args.max_evts, dtype=int)
        
        # beam events
        beam_mask = sel_idx[file["light/events/data"]['trig_type'][sel_idx] == 1]
        beam_wvfms, beam_noises, beam_baselines, beam_max_values, beam_clipped, beam_negs = get_waveform_info(
            file["light/wvfm/data"]['samples'], args.units, mask=beam_mask, ths=ptps)
        nbeam_evts = beam_wvfms.shape[0]
        
        # self-trigger events
        strig_mask = sel_idx[file["light/events/data"]['trig_type'][sel_idx] == 0]
        strig_wvfms, strig_noises, strig_baselines, strig_max_values, strig_clipped, strig_negs = get_waveform_info(
            file["light/wvfm/data"]['samples'], args.units, mask=strig_mask, ths=ptps)
        nstrig_evts = strig_wvfms.shape[0]
        
        # all events
        wvfms, noises, baselines, max_values, clipped, negs = get_waveform_info(
            file["light/wvfm/data"]['samples'], args.units, mask=sel_idx, ths=ptps)
        nevts = wvfms.shape[0]
        
        print("    Number of beam events:", nbeam_evts)
        print("    Number of self-trigger events:", nstrig_evts)
        print("    Total number of events:", nevts)

        ### DQM PLOTS ###

        # identify event with largest total integral
        integrals = np.sum(wvfms, axis=(1,2,3))
        max_integral_evt = np.argmax(integrals)
        evts = np.array([max_integral_evt])
        for evt in evts:
            plot_sum_waveform(wvfms - baselines[:, :, :, np.newaxis],
                              args.units, i_evt=evt, output_name='plot1_sumwvfm.pdf')
        print(f"Sum waveform plotted for file: {filename}")

        # noise spectra
        max_mask = get_max_value_mask(
            max_values[:args.powspec_nevts], ptps
        )
        #print("    Calculating noise spectra ({} events)".format(args.powspec_nevts))
        wvfms_v = adc16_to_voltage(file["light/wvfm/data"]['samples'][:args.powspec_nevts])
   
        freq_bins, noise_spectra, noise_spectrum, upper, lower = get_noise_spectra(
            wvfms_v, max_mask
        )
        rois_mhz = np.array([0.5, 1.8, 4.6, 7.1, 8.5, 10, 11.5, 19, 20, 25, 30])
        # convert to index from frequency bins
        rois_bins = np.array([np.argmin(np.abs(freq_bins*1e-6 - roi)) for roi in rois_mhz])

        # loop over rois, and save spectrum bin
        for i_roi in range(len(rois_bins)):
            roi_bin = rois_bins[i_roi]
            spur = noise_spectrum[:, :, roi_bin]
            # convert to string and replace '.' with '_'
            roi_mhz = str(rois_mhz[i_roi]).replace('.', '_')
            if args.write_json_blobs: save_spectra_as_json(
                i_file, spur, args.output_dir,
                f'noise_spectra_roi_{roi_mhz}mhz.json'
            )
        del wvfms_v
        plot_noise_spectra_epcb(
            freq_bins, noise_spectrum, None, None,
            skip_bad_channels=True, nevts=args.powspec_nevts,
            output_name='plot5_powspec.pdf',
        )
        #plot_noise_spectra_channels(
        #    freq_bins, noise_spectrum, None, None,
        #    skip_bad_channels=True, nevts=args.powspec_nevts,
        #    output_name='plot_0.pdf'
        #)
        print(f"Noise spectra calculated for file: {filename}")

        # Plot noise
        prev_noises = read_from_json(
            ncomps, args.output_dir, 'noises.json'
        ) if i_file > 0 else None
        noise_c, noise_l, noise_u = plot_noises(prev_noises,
            strig_noises, i_evt=np.arange(0, strig_baselines.shape[0], 1),
            mask_inactive=False, output_name='plot4_noises.pdf'
        )
        if args.write_json_blobs: save_as_json(i_file, noise_c, noise_l, noise_u,
                     args.output_dir, 'noises.json')

        # Plot baselines
        prev_baselines = read_from_json(
            ncomps, args.output_dir, 'baselines.json'
        ) if i_file > 0 else None
        bline_c, bline_l, bline_u = plot_baselines(prev_baselines,
            strig_baselines, i_evt=np.arange(0, strig_baselines.shape[0], 1),
            mask_inactive=False, output_name='plot2_baselines.pdf'
        )
        if args.write_json_blobs: save_as_json(i_file, bline_c, bline_l, bline_u,
                     args.output_dir, 'baselines.json')
        print(f"Noise and baselines plotted for file: {filename}")

      
        # Plot clipped fraction for total
        if nbeam_evts:
            # beam trigger
            prev_beam_clipped_inputs = read_eff_from_json(
                ncomps, args.output_dir, 'clipped_epcb_beam.json'
            ) if i_file > 0 else None
            prev_beam_clipped = clopper_pearson(
                prev_beam_clipped_inputs[0], prev_beam_clipped_inputs[1]
            ) if prev_beam_clipped_inputs is not None else None
            clip_pass, clip_tot = plot_clipped_epcb_fraction(
                prev_beam_clipped, beam_clipped, beam_max_values, ptps,
                "Beam trigger (% of events with clipped waveforms, normalized per ECPB)", output_name='plot6_clipped_beam_epcb.pdf'
            )
            if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                            args.output_dir, 'clipped_epcb_beam.json')
                        # plot clipped fraction for events with light on the channel itself
            
            if args.plot_all_clipped:
                            # beam events
                prev_beam_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_total_beam.json'
                ) if i_file > 0 else None
                prev_beam_clipped = clopper_pearson(
                    prev_beam_clipped_inputs[0], prev_beam_clipped_inputs[1]
                ) if prev_beam_clipped_inputs is not None else None
                clip_pass, clip_tot = plot_clipped_fraction(
                    prev_beam_clipped, beam_clipped,
                    title='Beam trigger (% of events with clipped waveforms)', output_name='plot6_clipped_beam.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_total_beam.json')
                # Plot clipped fraction for events with light on the channels' TPC
                # beam trigger
                prev_beam_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_tpc_beam.json'
                ) if i_file > 0 else None
                prev_beam_clipped = clopper_pearson(
                    prev_beam_clipped_inputs[0], prev_beam_clipped_inputs[1]
                ) if prev_beam_clipped_inputs is not None else None
                clip_pass, clip_tot = plot_clipped_tpc_fraction(
                    prev_beam_clipped, beam_clipped, beam_max_values, ptps,
                    title='Beam trigger (% of events with clipped waveforms, normalized per TPC)', output_name='plot6_clipped_beam_tpc.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_tpc_beam.json')
                            # Plot clipped fraction for events with light on the channels' EPCB

                # beam trigger
                prev_beam_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_ch_beam.json'
                ) if i_file > 0 else None
                prev_beam_clipped = clopper_pearson(
                    prev_beam_clipped_inputs[0], prev_beam_clipped_inputs[1]
                ) if prev_beam_clipped_inputs is not None else None
                clip_pass, clip_tot = plot_clipped_ch_fraction(
                    prev_beam_clipped, beam_clipped, beam_max_values, ptps,
                    "Beam trigger (% of events with clipped waveforms, normalized per channel)", output_name='plot6_clipped_ch_beam.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_ch_beam.json')
        if nstrig_evts:
           # self-trigger
            prev_strig_clipped_inputs = read_eff_from_json(
                ncomps, args.output_dir, 'clipped_epcb_self.json'
            ) if i_file > 0 else None
            prev_strig_clipped = clopper_pearson(
                prev_strig_clipped_inputs[0], prev_strig_clipped_inputs[1]
            ) if prev_strig_clipped_inputs is not None else None
            clip_pass, clip_tot = plot_clipped_epcb_fraction(
                prev_strig_clipped, strig_clipped, strig_max_values, ptps,
                "Self-trigger (% of events with clipped waveforms, normalized per EPCB)", output_name='plot_clipped6_self_epcb.pdf'
            )
            if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                            args.output_dir, 'clipped_epcb_self.json')
            print(f"Clipped EPCB fraction plotted for file: {filename}")
        
            if args.plot_all_clipped:
                # self-trigger
                prev_strig_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_total_self.json'
                ) if i_file > 0 else None
                prev_strig_clipped = clopper_pearson(
                    prev_strig_clipped_inputs[0], prev_strig_clipped_inputs[1]
                ) if prev_strig_clipped_inputs is not None else None
                clip_pass, clip_tot  = plot_clipped_fraction(
                    prev_strig_clipped, strig_clipped,
                    title='Self-trigger (% of events with clipped waveforms)', output_name='plot6_clipped_self.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_total_self.json')
                print(f"Clipped fraction plotted for file: {filename}")
                # self-trigger
                prev_strig_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_tpc_self.json'
                ) if i_file > 0 else None
                prev_strig_clipped = clopper_pearson(
                    prev_strig_clipped_inputs[0], prev_strig_clipped_inputs[1]
                ) if prev_strig_clipped_inputs is not None else None
                clip_pass, clip_tot = plot_clipped_tpc_fraction(
                    prev_strig_clipped, strig_clipped, strig_max_values, ptps,
                    title='Self-trigger (% of events with clipped waveforms, normalized per TPC)', output_name='plot6_clipped_self_tpc.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_tpc_self.json')
                print(f"Clipped TPC fraction plotted for file: {filename}")
    
                # self-trigger
                prev_strig_clipped_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'clipped_ch_self.json'
                ) if i_file > 0 else None
                prev_strig_clipped = clopper_pearson(
                    prev_strig_clipped_inputs[0], prev_strig_clipped_inputs[1]
                ) if prev_strig_clipped_inputs is not None else None
                clip_pass, clip_tot = plot_clipped_ch_fraction(
                    prev_strig_clipped, strig_clipped, strig_max_values, ptps,
                    "Self-trigger Trigger  (% of events with clipped waveforms, normalized per channel)", output_name='plot_clipped8_ch_self.pdf'
                )
                if args.write_json_blobs: save_eff_as_json(i_file, clip_pass, clip_tot,
                                args.output_dir, 'clipped_ch_self.json')
                print(f"Clipped channel fraction plotted for file: {filename}")

    
        if nbeam_evts:
            # plot negative spike fraction for events with light on the channels' TPC
            # beam trigger
            prev_beam_negs_inputs = read_eff_from_json(
                ncomps, args.output_dir, 'negatives_tpc_beam.json'
            ) if i_file > 0 else None
            prev_beam_negs = clopper_pearson(
                prev_beam_negs_inputs[0], prev_beam_negs_inputs[1]
            ) if prev_beam_negs_inputs is not None else None
            negs_pass, negs_tot = plot_neg_tpc_fraction(
                prev_beam_negs, beam_negs, beam_max_values, ptps,
                "Beam trigger (% of events with -ve spikes, normalized per TPC)", "plot6_negatives_beam_tpc.pdf"
            )
            if args.write_json_blobs: save_eff_as_json(i_file, negs_pass, negs_tot,
                            args.output_dir, 'negatives_tpc_beam.json')
                # plot negative spike fraction for events with lig`-
                # n the channels' EPCB
                # beam trigger
            if args.plot_all_negatives:
                prev_beam_negs_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'negatives_epcb_beam.json'
                ) if i_file > 0 else None
                prev_beam_negs = clopper_pearson(
                    prev_beam_negs_inputs[0], prev_beam_negs_inputs[1]
                ) if prev_beam_negs_inputs is not None else None
                negs_pass, negs_tot = plot_neg_epcb_fraction(
                    prev_beam_negs, beam_negs, beam_max_values, ptps,
                    "Beam trigger (% of events with -ve spikes, normalized per EPCB)", "plot6_negatives_beam_epcb.pdf"
                )
                if args.write_json_blobs: save_eff_as_json(i_file, negs_pass, negs_tot,
                                args.output_dir, 'negatives_epcb_beam.json')

        if nstrig_evts:
            # self-trigger
            prev_strig_negs_inputs = read_eff_from_json(
                ncomps, args.output_dir, 'negatives_tpc_self.json'
            ) if i_file > 0 else None
            prev_strig_negs = clopper_pearson(
                prev_strig_negs_inputs[0], prev_strig_negs_inputs[1]
            ) if prev_strig_negs_inputs is not None else None
            negs_pass, negs_tot = plot_neg_tpc_fraction(
                prev_strig_negs, strig_negs, strig_max_values, ptps,
                "Self-trigger (% of events with -ve spikes, normalized per TPC)", "plot6_negatives_self_tpc.pdf"
            )
            if args.write_json_blobs: save_eff_as_json(i_file, negs_pass, negs_tot,
                            args.output_dir, 'negatives_tpc_self.json')
            print(f"Negative spike TPC fraction plotted for file: {filename}")
            if args.plot_all_negatives:
                # self-trigger
                prev_strig_negs_inputs = read_eff_from_json(
                    ncomps, args.output_dir, 'negatives_epcb_self.json'
                ) if i_file > 0 else None
                prev_strig_negs = clopper_pearson(
                    prev_strig_negs_inputs[0], prev_strig_negs_inputs[1]
                ) if prev_strig_negs_inputs is not None else None
                negs_pass, negs_tot = plot_neg_epcb_fraction(
                    prev_strig_negs, strig_negs, strig_max_values, ptps,
                    "Self-trigger  (% of events with -ve spikes, normalized per EPCB)", "plot6_negatives_self_epcb.pdf"
                )
                if args.write_json_blobs: save_eff_as_json(i_file, negs_pass, negs_tot,
                                args.output_dir, 'negatives_epcb_self.json')
                print(f"Negative spike EPCB fraction plotted for file: {filename}")

        ### GRAFANA PLOTS ###

        # check for flatlining channels
        flatlined = check_flatline(
            max_values, threshold=0.1
        )
        # plotting flatlined channels
        plot_flatline_mask(
            flatlined, cs, output_name=f'light_dqm_flatline_{args.file_syntax}{args.start_run}.pdf',
            times = (start_central, end_central)
        )
        # checking for baseline fluctuations
        baselined = check_baseline(
            prev_baselines, (bline_c, bline_l, bline_u), units=args.units,
            threshold = 1000
        )
        # plotting baseline fluctuations
        plot_baseline_mask(
            baselined, cs, output_name=f'light_dqm_baseline_{args.file_syntax}{args.start_run}.pdf',
            times = (start_central, end_central)
        )

        # search for files in the output directory and merge pdfs
        merger = PdfMerger()
        merger_grafana = PdfMerger()
        # Create a PDF with the arguments, file index, and unix timestamp
        input_path_lines = []
        max_line_length = 40
        input_path = args.input_path
        while len(input_path) > max_line_length:
            split_idx = input_path.rfind('/', 0, max_line_length)
            if split_idx == -1:
                split_idx = max_line_length
            input_path_lines.append(input_path[:split_idx])
            input_path = input_path[split_idx:]
        input_path_lines.append(input_path)
       

        args_list = [
            f"File index: {i_file}",
            f"--input path: {input_path_lines}",
            "\n",
            f"Data start timestamp (CT): {start_central}",
            f"Data end timestamp   (CT): {end_central}",
            f"DQM runtime          (CT): {(time.time() - start_time):.2f} seconds"
            "\n",
            f"--nfiles {args.nfiles}",
            f"--start_run {args.start_run}",
            f"--ncomp {args.ncomp}",
            "\n",
            f"--output_dir {args.output_dir}",
            f"--units {args.units}",
            f"--ptps16bit {args.ptps16bit}",
            "\n",
            f"--max_evts {args.max_evts}",
            f"--powspec_nevts {args.powspec_nevts}"
        ]
        args_pdf = os.path.join(args.tmp_dir, f"args_list_{args.start_run}.pdf")
        with PdfPages(args_pdf) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.axis('off')
            text = "\n".join(args_list)
            ax.text(0.01, 0.99, text, va='top', ha='left', fontsize=12, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)
        # Insert the args PDF at the top of the merged file
        merger.append(args_pdf)
        merger_grafana.append(args_pdf)
        os.remove(args_pdf)

        plot_files = sorted(glob.glob(os.path.join(args.tmp_dir, "plot*.pdf")))

        for plot_path in plot_files:
            if os.path.exists(plot_path):
                merger.append(plot_path)
                os.remove(plot_path)

        merged_pdf_path = os.path.join(args.output_dir, f"light_dqm_main_{args.file_syntax}{args.start_run}.pdf")
        merger.write(merged_pdf_path)
        merger.close()

        if args.merge_grafana_plots:
            # Append all Grafana plot PDFs to the merger
            grafana_plotnames = sorted(
                glob.glob(os.path.join(args.output_dir_dir, "light_dqm_baseline*.pdf"))
                + glob.glob(os.path.join(args.output_dir_dir, "light_dqm_flatline*.pdf"))
            )
            for grafana_plotname in grafana_plotnames:
                grafana_plot_path = f"{args.output_dir}{grafana_plotname}"
                if os.path.exists(grafana_plot_path):
                    merger_grafana.append(grafana_plot_path)
                    os.remove(grafana_plot_path)
            merged_grafana_pdf_path = f"{args.output_dir}merged_grafana_plots_{args.start_run}.pdf"
            merger_grafana.write(merged_grafana_pdf_path)
            merger_grafana.close()
            print(f"Merged Grafana plots saved to: {merged_grafana_pdf_path}")

        # timing for file processing
        print(f"Processing completed for file: {filename}")
        print(f"Time taken for file {i_file}: {time.time() - file_time:.2f} seconds")
        file_time = time.time()
    if not proc_files:
        raise ValueError("None of the files were found")
    footer()
    print(f"Time taken all files: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
