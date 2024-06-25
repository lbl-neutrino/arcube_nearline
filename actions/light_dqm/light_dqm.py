#!/usr/bin/env python3

##########################################
##                                      ##
##            ~ Light DQM ~             ##
##                                      ##
##########################################
##
##  Written 23.02.2024
##
##    - Angela White: ajwhite@uchicago.edu
##    - Livio Calivers: <email>
##
##  Version:  v02, 23.06.2024, direct questions to A. J. White
##  Previous: v01, 23.02.2024
## 
##  Goal: 
##    - Monitor Quality of light data after
##      ADC64 flow step.
##    - Output plots to database
##
###########################################

## import packages ##
import numpy as np
import matplotlib
import h5py, glob, argparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.fft import rfft, rfftfreq
import awkward as ak
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
import matplotlib.patches as mpatches
from scipy.signal import convolve
from scipy.signal import find_peaks
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator)
from matplotlib.colors import LogNorm
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import emoji
import json


_old_axes_init = Axes.__init__


def _new_axes_init(self, *a, **kw):
    _old_axes_init(self, *a, **kw)
    # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
    # 3 => leave text and legends vectorized
    self.set_rasterization_zorder(3)


def rasterize_plots():
    Axes.__init__ = _new_axes_init
    
rasterize_plots()
    
## define system constants ##
SAMPLE_RATE = 0.016 # us
BIT = 4  # factor from unused ADC bits on LRS: would be nice to have in a resource .yaml
PRE_NOISE = 90 # Will need to be re-defined once we know beam timing
THRESHOLD = 600 # For the FFT, used to define events with signal peaks
## If single module data: could use this for labeling ##
MOD = 0
## Some sort of saturation value re:Livio ##
sat = 32767
SAMPLES = 1000
    
sipm_channels = ([4,5,6,7,8,9] + \
                 [10,11,12,13,14,15] + \
                 [20,21,22,23,24,25] + \
                 [26,27,28,29,30,31] + \
                 [36,37,38,39,40,41] + \
                 [42,43,44,45,46,47] + \
                 [52,53,54,55,56,57] + \
                 [58,59,60,61,62,63])

## Plotting Functions ##

## Function for Baseline ##
## 
def emoji_thang(bsln_arr, input1, input2, module, typ):
    length = 64
    expected = np.ones(length)
    inactive = input1
    special_value = 0
    expected[inactive] = special_value
    bbase = input2[f'{module * 2 + typ}']
    expected[bbase] = special_value
    expected = np.int64(expected)
    baseline = bsln_arr[module * 2 + typ]
    baseline[inactive] = special_value
    baseline[bbase] = special_value

    if np.array_equal(baseline, expected) == True:
        if np.sum(baseline, dtype=int) == 48:
            color = 'limegreen'
            text = emoji.emojize(':smile:',language='alias')
        else:
            color = 'deeppink'
            text = emoji.emojize(':winking_face:',language='alias')  
            print('ee')
    else:
        color = 'orangered'
        text = emoji.emojize(':angry:',language='alias')
        print('ff')
    
    return color, text
##
##
def baseline_graf(bsln_arr, MODULES, mod, input1, input2, output):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if MODULES==1:
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
        for j in range(2):
            color = 'limegreen' if bsln_arr[i * 2 + j] == 48 else 'deeppink'
            axs[j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
            axs[j].set_xticks([])
            axs[j].set_yticks([])
            axs[j].annotate('{}'.format(current_time), xy=(0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=8)
            text = emoji.emojize(':smile:',language='alias') if bsln_arr[i * 2 + j] == 48 else emoji.emojize(':winking_face:',language='alias')
            axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

        for iax in axs.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(16))
            iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
            iax.grid(axis = 'x',which="both")

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        axs[0].set_xlabel("LRS Baseline")
        axs[1].set_xlabel("LRS Baseline")
        plt.savefig(output)
        plt.close()
        
    else:
        fig, axs = plt.subplots(4, 2, figsize=(4, 8))
        for i in range(4):
            for j in range(2):
                color, text = emoji_thang(bsln_arr, input1, input2, i, j)
                # Plot rectangle
                axs[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].annotate('{}'.format(current_time), xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)
                axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

            for iax in axs.reshape(-1):
                iax.xaxis.set_major_locator(MultipleLocator(16))
                iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
                iax.grid(axis = 'x',which="both")

            row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
            col_headers=["ACL","LCM"]
            font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
            add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

            axs[3,0].set_xlabel("LRS Baseline")
            axs[3,1].set_xlabel("LRS Baseline")
        plt.savefig(output)
        plt.close()
##
##
def bsline_pFile(wvfm_all, mod, output1, output2, MODULES, inputs):
    
    max_amp_pchan_acl = []
    max_amp_pchan_lcm = []
    
    baseline_list = []
    
    if MODULES==1:
        
        max_amp_pchan_acl.append(wvfm_all[:,0,:,:50].mean(axis=-1).mean(axis=0))
        max_amp_pchan_lcm.append(wvfm_all[:,1,:,:50].mean(axis=-1).mean(axis=0))

        #fig, ax = plt.subplots(1,2,figsize=(8, 2))
        fig, ax = plt.subplots(1,2,figsize=(16, 4))
        ax[0].plot(max_amp_pchan_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pchan_lcm[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Waveform Baseline [ADC]")
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(63))
            iax.xaxis.set_minor_locator(FixedLocator([3.5,9.5,15.5,19.5,25.5,31.5,35.5,41.5,47.5,51.5,57.5]))
            iax.grid(axis = 'x',which="both")
            #iax.set_ylim(-1000,sat+1000)

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[0].set_xlabel("ACL")
        ax[1].set_xlabel("LCM") 
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output1.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pchan_acl.append(wvfm_all[:,mod*2,:,:50].mean(axis=-1).mean(axis=0))
            max_amp_pchan_lcm.append(wvfm_all[:,mod*2+1,:,:50].mean(axis=-1).mean(axis=0))

        fig, ax = plt.subplots(4,2,figsize=(16,16))
        for mod in range(4):
            ax[mod,0].plot(max_amp_pchan_acl[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,1].plot(max_amp_pchan_lcm[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,0].set_ylabel("Baseline")
            
            baseline_list.append(max_amp_pchan_acl[mod])
            baseline_list.append(max_amp_pchan_lcm[mod])
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(63))
            iax.xaxis.set_minor_locator(FixedLocator([3.5,9.5,15.5,19.5,25.5,31.5,35.5,41.5,47.5,51.5,57.5]))
            iax.grid(axis = 'x',which="both")
            #iax.set_ylim(-1000,sat+1000)

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("ACL")
        ax[3,1].set_xlabel("LCM")
        txt = ' .'
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output1.savefig()
        plt.close()
        
    baseline_array = np.array(baseline_list)
    baseline_mask = (baseline_array > -31000) & (baseline_array < -25000)
    baseline_test = np.int64(baseline_array*(baseline_mask==0)+baseline_mask)
    baseline_test[baseline_test != 1] = 0
    baseline_graf(baseline_test, MODULES, mod, inputs['inactive_channels'], inputs['bad_baseline'], output2)
    
##
##
def bsline_drift(wvfm_all, mod, output, MODULES):
    
    max_amp_pchan_acl = []
    max_amp_pchan_lcm = []
    
    if MODULES==1:
        
        max_amp_pchan_acl.append(wvfm_all[:,0,:,:50].mean(axis=-1))
        max_amp_pchan_lcm.append(wvfm_all[:,1,:,:50].mean(axis=-1))

        fig, ax = plt.subplots(1,2,figsize=(8,2))
        ax[0].plot(max_amp_pchan_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pchan_lcm[1],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Baseline")
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(47))
            iax.xaxis.set_minor_locator(FixedLocator([3.5,9.5,15.5,19.5,25.5,31.5,35.5,41.5,47.5,51.5,57.5]))
            iax.grid(axis = 'x',which="both")

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[0].set_xlabel("ACL")
        ax[1].set_xlabel("LCM") 
        output.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pchan_acl.append(wvfm_all[:,mod*2,sipm_channels,:50].mean(axis=-1))
            max_amp_pchan_lcm.append(wvfm_all[:,mod*2+1,sipm_channels,:50].mean(axis=-1))
            
            cumulative_sum_acl = np.cumsum(np.array(max_amp_pchan_acl), axis=1)
            cumulative_sum_lcm = np.cumsum(np.array(max_amp_pchan_lcm), axis=1)
            num_elements = np.arange(1,np.array(max_amp_pchan_acl).shape[1] + 1)
            cumulative_average_acl = cumulative_sum_acl / num_elements.reshape(-1, 1)
            cumulative_average_lcm = cumulative_sum_lcm / num_elements.reshape(-1, 1)

        fig, ax = plt.subplots(4,2,figsize=(8,8))
        for mod in range(4):
            acl = ax[mod,0].pcolormesh(cumulative_average_acl[mod], vmin=-3e4, vmax=3e4, cmap='viridis')
            lcm = ax[mod,1].pcolormesh(cumulative_average_lcm[mod], vmin=-3e4, vmax=3e4, cmap='viridis')
            ax[mod,0].set_ylabel("Event")
            num_bins_y, num_bins_x = cumulative_average_acl[mod].shape
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(6))
            iax.grid(axis = 'x',which="both")

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("Active Channel")
        ax[3,1].set_xlabel("Active Channel")
        cbar_ax = fig.add_axes([0.95, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(lcm, cax=cbar_ax, label=r'Charge [$10^3$] e')
        cbar.set_label(r'Baseline Value', size=12)
        cbar_ax.tick_params(labelsize=12)
        output.savefig()
        plt.close()
##
##

def trigger_timing(light_wvfms, events, output):
    
    adc_list = np.arange(np.shape(light_wvfms)[1])
    
    for adc in adc_list:

        transposed_wvfms = np.transpose(light_wvfms, (1, 2, 0, 3))[adc,:,:,:]
    
        fig, ax = plt.subplots(4, 2, figsize=(11.5, 7.5), sharex=True, layout='constrained')
        for summ in range(8):
            start = summ*6
            end = start+6
            flattened_wvfms = transposed_wvfms[start:end, :, :]
            sum_wvfm = np.clip(np.sum(flattened_wvfms, axis=0),None,sat)

            chan = np.shape(flattened_wvfms)[0]
            for ev in events:
                #print(ev)
                ax[summ//2, summ%2].plot(np.linspace(0,SAMPLES,SAMPLES), sum_wvfm[ev, :], color='grey', linewidth=0.8, label='Sum')
                for i in range(chan):
                    ax[summ//2, summ%2].plot(np.linspace(0,SAMPLES,SAMPLES), flattened_wvfms[i,ev,:], linewidth=0.8)
            ax[summ//2, summ%2].set_title('ADC '+str(adc)+': Channels ['+str(start)+':'+str(end-1)+']')
            ax[summ//2, summ%2].set_ylim(-100,33000)
            ax[summ//2, summ%2].grid(True)
            ax[summ//2, summ%2].set_ylim(0, 35000)
            ax[summ//2, 0].set_ylabel('ADC Value')
            ax[3, summ%2].set_xlabel('Sample [0.016 μs]')

        plt.grid(True)
        rect = plt.Rectangle((0, 0), 1200, 800, fill=False, edgecolor='black', lw=2)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        plt.legend(unique.values(), unique.keys())
        fig.patches.append(rect)
        output.savefig()
        plt.close()

##
##
def adjacent_values(arr, color, label, marker, ypt):
    listt = []
    for i in range(len(arr) - 1):
        if abs(arr[i] - arr[i+1]) > 1:
            listt.append(arr[i])
            listt.append(arr[i+1])
            #print('list len:', len(listt))
    try:
        min_val = np.min(np.array(listt))
        max_val = np.max(np.array(listt))
        #print(min_val, max_val)
        plt.scatter(min_val, ypt, color=color, marker=marker, label=str(label))
        plt.scatter(max_val, ypt, color=color, marker=marker)
    except:
        #print('no peaks')
        a=0
##
##
def signal_region(wvfm_all, output, skip_int):
    
    adc_list = np.arange(np.shape(wvfm_all)[1])
    
    fig = plt.figure(figsize=(12,3))
    labels = ['ACL 0','LCM 1','ACL 2','LCM 3','ACL 4','LCM 5','ACL 6','LCM 7']
    colors = ['navy','maroon','blue','crimson','dodgerblue','tomato','lightblue','orange']
    markers = ['3','4','3','4','3','4','3','4']
    adder = [2,4,2,4,2,4,2,4]
    
    for adc in adc_list:
        
        transposed_wvfms = np.transpose(wvfm_all, (1, 2, 0, 3))[adc,:,::skip_int,:]
        
        for summ in range(8):
            start = summ*6
            end = start+6
            flattened_wvfms = np.concatenate(transposed_wvfms[start:end, :, :], axis=0)
            uber_flattened_wvfms = np.concatenate(flattened_wvfms, axis=0)
            x_hist_portion = np.tile(np.arange(1000), np.shape(flattened_wvfms)[0])
            
            hist = np.histogram2d(x_hist_portion, uber_flattened_wvfms, bins=(1000,100))

            non_noise = hist[0][:,3:]
            matrix_mask = (non_noise > 2)
            mask_sum = np.sum(matrix_mask, axis=1)
            
            window_size = 20
            y_smooth = np.convolve(mask_sum/window_size, np.ones(window_size)/window_size, mode='valid')

            adjacent_values(np.where(y_smooth*window_size == 0)[0], colors[adc], labels[adc], markers[adc], start+adder[adc])

    plt.xlim(0,1000)
    plt.ylim(0,48)
    handles, labels = plt.gca().get_legend_handles_labels()

    # Include every fourth label
    every_fourth_labels = labels[::4]
    every_fourth_handles = handles[::4]

    # Plot legend with every fourth label
    plt.legend(every_fourth_handles, every_fourth_labels, loc='upper center', fontsize=11)
    plt.grid(True)
    plt.xlabel('Sample [0.016 μs]', fontsize=14)
    plt.ylabel('Active Channel', fontsize=14)
    plt.title('(LRS Waveform) Beam Response Window', fontsize=16)
    output.savefig()
    plt.close()
##
## Functions for FFTs: ##
##
def noise_datasets(no_ped_adc, THRESHOLD):

    adc_signal_indices=[]
    for i in range(0,len(no_ped_adc),2):
        if max(no_ped_adc[i])<THRESHOLD:
            adc_signal_indices.append(i)
        else:
            pass
    
    adc_normal_pretrig = []
    for i in adc_signal_indices:
        waveform = (no_ped_adc[i][0:PRE_NOISE])
        adc_normal_pretrig.append(np.array(waveform))
        if len(adc_normal_pretrig)>500:
            break
    adc_normal_pretrig = np.array(adc_normal_pretrig[0:500])

    # Calculate power spectra using FFT
    freqs = np.fft.fftfreq(PRE_NOISE, SAMPLE_RATE)
    freqs = freqs[:PRE_NOISE//2] # keep only positive frequencies
    freq_matrix = np.tile(np.array(freqs), (len(adc_normal_pretrig),1))
    frequencies = np.ndarray.flatten(np.array(freq_matrix))

    psds = []
    for wave in adc_normal_pretrig:
        spectrum = np.fft.rfft(wave)
        psd = np.abs(spectrum[:PRE_NOISE//2])**2 / (PRE_NOISE * SAMPLE_RATE)
        psd[1:] *= 2 # double the power except for the DC component
        psds.append(psd)

    ref = 1 #(everything is in integers?)
    power = np.ndarray.flatten(np.array(psds))
    p_dbfs = 20 * np.log10(power/ref)
    
    del power
    del psds
    
    return adc_signal_indices, frequencies, adc_normal_pretrig, p_dbfs
##
##
def power_hist_maxes(adc_dataset):
    adc_freq = adc_dataset[1]
    adc_pdbfs = adc_dataset[3]
    hist, *edges = np.histogram2d(adc_freq[(adc_pdbfs)>-500], adc_pdbfs[(adc_pdbfs)>-500], bins=32)
    ycenters = (edges[1][:-1] + edges[1][1:]) / 2
    xcenters = (edges[0][:-1] + edges[0][1:]) / 2

    maxes = []
    for array in hist:
        maxes.append(np.where(array == max(array))[0][0])
    max_bins = [ycenters[i] for i in maxes]
    
    del adc_freq
    del adc_pdbfs

    return xcenters, max_bins
##
##
def power_spec_plots(adc0_dataset, adc0_max, adc1_dataset, adc1_max, CUTOFF, mod, output): 
    fig = plt.figure(figsize=(16,4))
    x = np.linspace(0,CUTOFF-1,CUTOFF)
    if len(adc0_dataset[2]) > 0:
        y0 = adc0_dataset[2][10]/BIT
        plt.plot(x, y0, "-", color='green', label='ACL')
    if len(adc1_dataset[2]) > 0:
        y1 = adc1_dataset[2][10]/BIT
        plt.plot(x, y1, "-", color='yellowgreen', label='LCM')
    plt.title('Waveform Example : Module '+str(mod), fontsize=16)
    plt.xlabel(r'Time Sample [0.016 $\mu$s]', fontsize=12)
    plt.ylabel('SiPM Channel Output', fontsize=12)
    plt.legend()
    output.savefig()
    plt.close()
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 10))
    adc0_freq = adc0_dataset[1]
    adc0_pdbfs = adc0_dataset[3]
    adc1_freq = adc1_dataset[1]
    adc1_pdbfs = adc1_dataset[3]
    
    if len(adc0_pdbfs)>0:
        hist1 = ax[0].hist2d(adc0_freq[adc0_pdbfs>-500], adc0_pdbfs[adc0_pdbfs>-500], bins=32, \
                            norm=mpl.colors.LogNorm(), cmap='viridis')
        fig.colorbar(hist1[3], ax=ax, location='bottom')
        ax[0].plot(adc0_max[0],adc0_max[1],'o-k')
    ax[0].set_title('ACL Noise Power Spectrum')
    ax[0].set_ylim(-110,230)
    ax[0].set_xlabel('Frequency [MHz]',fontsize=14)
    if len(adc1_pdbfs>0):
        hist2 = ax[1].hist2d(adc1_freq[adc1_pdbfs>-500], adc1_pdbfs[adc1_pdbfs>-500], bins=32, \
                            norm=mpl.colors.LogNorm(), cmap='viridis')
        ax[1].plot(adc1_max[0],adc1_max[1],'o-k')
    ax[1].set_title('LCM Noise Power Spectrum')
    ax[1].set_ylim(-110,230)
    ax[1].set_xlabel('Frequency [MHz]',fontsize=14)
    ax[0].set_ylabel('Power Spectrum [dB]',fontsize=14)
    fig.suptitle('Module '+str(mod)+': Noise FFT, {} Waveforms\n'.format(len(adc1_dataset[2])), fontsize=16, x=0.5, y=0.95)
    # Show the plot
    plt.subplots_adjust(hspace=0.2, wspace=0.02, bottom = 0.35)
    txt = ' '
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    output.savefig()
    plt.close()
##
## End of FFT Functions ##
    
## Define Amplitude Plots ##    
##
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )
##
##
def amp_by_tick(adc_matrix, start, tick_start, tick_end, title, mod, color, output):
    
    tick_ray = np.arange(tick_start, tick_end, 1)
    
    amp_hist = []
    
    for j in range(48):
        amp = []
        l_chan_array = adc_matrix[j+start,:,tick_start:tick_end]
        for i_t, tick in enumerate(tick_ray): 
            array1 = l_chan_array[:,i_t]/BIT
            amp.append(np.max(abs(array1)))
        amp_hist.append(np.array(amp))
    amp_hist_arr = np.concatenate(amp_hist)
    tick_arr = np.tile(tick_ray, 48)

    fig = plt.figure(figsize=(16,6))
    
    xbins = np.linspace(0, 999, 500)
    ybins = np.linspace(0, 1500, 50)
    
    plt.hist2d(np.array(tick_arr), np.array(amp_hist_arr), bins=[xbins, ybins], cmap=color, norm=LogNorm())
    cbar = plt.colorbar()

    plt.xlabel('Tick',fontsize=14)
    plt.ylabel('Maximum Amplitude',fontsize=14)
    plt.title('Module '+str(mod)+': '+str(title)+'\nMax Amplitude Per Tick ['+str(tick_start)+':'+str(tick_end)+']',fontsize=16)
    plt.grid(True)
    txt = ' '
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    output.savefig()
    plt.close()
##
##   
def flow2sim(waveforms, function, start_tick, end_tick, mod):
    
    modules = np.int64(np.shape(waveforms)[1])
    
    if modules == 2: 
        adc_acl = function(waveforms[:,0,:,:], start_tick, end_tick, mod)
        adc_lcm = function(waveforms[:,1,:,:], start_tick, end_tick, mod)                
        mod_array = np.concatenate((adc_lcm[0:6],adc_acl[0:6],\
                                    adc_lcm[6:12],adc_acl[6:12],\
                                    adc_lcm[12:18],adc_acl[12:18],\
                                    adc_lcm[18:24],adc_acl[18:24],\
                                    adc_lcm[24:30],adc_acl[24:30],\
                                    adc_lcm[30:36],adc_acl[30:36],\
                                    adc_lcm[36:42],adc_acl[36:42],\
                                    adc_lcm[42:48],adc_acl[42:48]))
    
    else: 
        full_mod_array = []
        for i in range(np.int64(modules/2)):
            adc_acl = function(waveforms[:,i*2,:,:], start_tick, end_tick, mod)
            adc_lcm = function(waveforms[:,(i*2)+1,:,:], start_tick, end_tick, mod) 
            mod_only_array = np.concatenate((adc_lcm[0:6],adc_acl[0:6],\
                                             adc_lcm[6:12],adc_acl[6:12],\
                                             adc_lcm[12:18],adc_acl[12:18],\
                                             adc_lcm[18:24],adc_acl[18:24],\
                                             adc_lcm[24:30],adc_acl[24:30],\
                                             adc_lcm[30:36],adc_acl[30:36],\
                                             adc_lcm[36:42],adc_acl[36:42],\
                                             adc_lcm[42:48],adc_acl[42:48]))              
            full_mod_array.append(mod_only_array)
        mod_array = np.concatenate(np.array(full_mod_array))
    
    return mod_array
##
## Amplitude Plotting Functions ##
##
def maxAmp_pModpEv(wvfm_all, mod, output, MODULES):
    
    max_amp_pmod_acl = []
    max_amp_pmod_lcm = []
    
    if MODULES==1:
        
        max_amp_pmod_acl.append(wvfm_all[:,0,:,:].max(axis=-1).max(axis=-1))
        max_amp_pmod_lcm.append(wvfm_all[:,1,:,:].max(axis=-1).max(axis=-1))

        fig, ax = plt.subplots(1,2,figsize=(16,4))
        ax[0].plot(max_amp_pmod_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pmod_lcm[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Max amp [ADC]",fontfamily="monospace")
    
        for iax in ax.reshape(-1):
            #iax.xaxis.set_major_locator(MultipleLocator(24))
            iax.xaxis.set_major_locator(MultipleLocator(10))
            iax.grid(axis = 'x',which="both")
            iax.set_yscale('log')
            iax.set_ylim(10,sat*10)
            iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[0].set_xlabel("Event ID",fontfamily="monospace")
        ax[1].set_xlabel("Event ID",fontfamily="monospace") 
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pmod_acl.append(wvfm_all[:,mod*2,:,:].max(axis=-1).max(axis=-1))
            max_amp_pmod_lcm.append(wvfm_all[:,mod*2+1,:,:].max(axis=-1).max(axis=-1))

        fig, ax = plt.subplots(4,2,figsize=(16,16))
        for mod in range(4):
            ax[mod,0].plot(max_amp_pmod_acl[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,1].plot(max_amp_pmod_lcm[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,0].set_ylabel("Max Amp [ADC]",fontfamily="monospace")
    
        for iax in ax.reshape(-1):
            #iax.xaxis.set_major_locator(MultipleLocator(24))
            iax.xaxis.set_major_locator(MultipleLocator(10))
            iax.grid(axis = 'x',which="both")
            iax.set_yscale('log')
            iax.set_ylim(10,sat*10)
            iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("Event ID",fontfamily="monospace")
        ax[3,1].set_xlabel("Event ID",fontfamily="monospace")
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output.savefig()
        plt.close()
##
def maxamp_graf(mxamp_arr, MODULES, mod, output, input1, input2):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if MODULES==1:
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
        for j in range(2):
            color = 'limegreen' if mxamp_arr[i * 2 + j] != 0 else 'deeppink'
            axs[j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
            axs[j].set_xticks([])
            axs[j].set_yticks([])
            axs[j].annotate('{}'.format(current_time), xy=(0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=8)
            text = emoji.emojize(':smile:',language='alias') if mxamp_arr[i * 2 + j] != 0 else emoji.emojize(':winking_face:',language='alias')
            axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

        for iax in axs.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(16))
            iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
            iax.grid(axis = 'x',which="both")

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        axs[0].set_xlabel("LRS Dead Channels")
        axs[1].set_xlabel("LRS Dead Channels")
        plt.savefig(output)
        plt.close()
        
    else:
    # Loop through each subplot
        fig, axs = plt.subplots(4, 2, figsize=(4, 8))
        for i in range(4):
            for j in range(2):
                color, text = emoji_thang(mxamp_arr, input1, input2, i, j)
                # Plot rectangle
                axs[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].annotate('{}'.format(current_time), xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)
                axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

            for iax in axs.reshape(-1):
                iax.xaxis.set_major_locator(MultipleLocator(16))
                iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
                iax.grid(axis = 'x',which="both")
                #iax.set_ylim(-1000,sat+1000)

            row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
            col_headers=["ACL","LCM"]
            font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
            add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

            axs[3,0].set_xlabel("LRS Dead Channels")
            axs[3,1].set_xlabel("LRS Dead Channels")
        plt.savefig(output)
        plt.close()
##
def maxAmp_pChanpFile(wvfm_all, mod, output1, output2, MODULES, inputs):
    
    max_amp_pchan_acl = []
    max_amp_pchan_lcm = []
    
    deadchan_list = []
    
    if MODULES==1:        
        max_amp_pchan_acl.append(wvfm_all[:,0,:,:].max(axis=-1).max(axis=0))
        max_amp_pchan_lcm.append(wvfm_all[:,1,:,:].max(axis=-1).max(axis=0))

        fig, ax = plt.subplots(1,2,figsize=(16,4))
        ax[0].plot(max_amp_pchan_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pchan_lcm[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Max Amp [ADC]")
        
        deadchan_list.append(max_amp_pchan_acl[0][sipm_channels])
        deadchan_list.append(max_amp_pchan_lcm[1][sipm_channels])
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(63))
            iax.xaxis.set_minor_locator(FixedLocator([3.5,9.5,15.5,19.5,25.5,31.5,35.5,41.5,47.5,51.5,57.5]))
            iax.grid(axis = 'x',which="both")
            iax.set_yscale('log')
            iax.set_ylim(10,sat*10)
            iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[0].set_xlabel("Channel",fontfamily="monospace")
        ax[1].set_xlabel("Channel",fontfamily="monospace")    
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output1.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pchan_acl.append(wvfm_all[:,mod*2,:,:].max(axis=0).max(axis=-1))
            max_amp_pchan_lcm.append(wvfm_all[:,mod*2+1,:,:].max(axis=0).max(axis=-1))

        fig, ax = plt.subplots(4,2,figsize=(16,16))
        for mod in range(4):
            ax[mod,0].plot(max_amp_pchan_acl[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,1].plot(max_amp_pchan_lcm[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,0].set_ylabel("Max Amp [ADC]")
            
            deadchan_list.append(max_amp_pchan_acl[mod])
            deadchan_list.append(max_amp_pchan_lcm[mod])
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(63))
            iax.xaxis.set_minor_locator(FixedLocator([3.5,9.5,15.5,19.5,25.5,31.5,35.5,41.5,47.5,51.5,57.5]))
            iax.grid(axis = 'x',which="both")
            iax.set_yscale('log')
            iax.set_ylim(10,sat*10)
            #iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)
            iax.axhline(y = 1000, color = 'r', linestyle = '-', alpha=0.5)

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("Channel",fontfamily="monospace")
        ax[3,1].set_xlabel("Channel",fontfamily="monospace")
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output1.savefig()
        plt.close()
        
    deadchan_array = np.array(deadchan_list)
    deadchan_mask = (deadchan_array > 100)
    deadchan_test = np.int64(deadchan_mask)
    maxamp_graf(deadchan_test, MODULES, mod, output2, inputs['inactive_channels'], inputs['weird_channels'])
##
##
def hist_maxAmp(inpuT, tick_start, tick_end, mod):
    input_max = np.abs(inpuT[:,:,tick_start:tick_end]).max(axis=2)/BIT
    flipped_max = np.transpose(input_max)
    return flipped_max
##
##
def max_amp(wvfm_matrix, tick_start, tick_end, mod, output, MODULES):

    fig, ax = plt.subplots(figsize=(16, 8))
    
    color_array = ['yellow', 'orange', 'red', 'magenta']
    
    if MODULES==1:
        notflat_max = flow2sim(wvfm_matrix, hist_maxAmp, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_max)[-1])
        flat_max = np.concatenate(notflat_max, axis=0)
        flat_channel = np.arange(0,96,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, color_array[mod], 0.2, 'Mod '+str(mod))]
        for i in range(0, 96, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
            
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        
        x_range = (0, 96)
        y_range = (0, np.max(flat_max)*1.1)
        hist1 = ax.hist2d(flat_channels, flat_max, bins=(96,100), range=[x_range, y_range], norm=mpl.colors.LogNorm(), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none',
                         alpha=0.7,lw=1, label='Dead: '+str(zero_y_bins[i])))
        
        ax.set_title('Single Module: Maximum Amplitude by Channel', fontsize=16)
        
    else: 
        notflat_max = flow2sim(wvfm_matrix, hist_maxAmp, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_max)[-1])
        flat_max = np.concatenate(notflat_max, axis=0)
        flat_channel = np.arange(0,384,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, 'yellow', 0.2, 'Mod 0'),
                   (96, 192, 'orange', 0.3, 'Mod 1'),
                   (192, 288, 'red', 0.2, 'Mod 2'),
                   (288, 384, 'magenta', 0.2, 'Mod 3'),]
        for i in range(0, 384, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
        
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        ax.axvline(x=144, color='black', linestyle=':')
        ax.axvline(x=192, color='black', linestyle=':')
        ax.axvline(x=240, color='black', linestyle=':')
        ax.axvline(x=288, color='black', linestyle=':')
        ax.axvline(x=336, color='black', linestyle=':')
        ax.axvline(x=384, color='black', linestyle=':')
        
        x_range = (0, 384)
        y_range = (0, np.max(flat_max)*1.1)
        
        hist1 = ax.hist2d(flat_channels, flat_max, bins=(384,100), range=[x_range, y_range], norm=mpl.colors.LogNorm(vmax=50), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none',
                         alpha=0.7, lw=1))
        
        ax.set_title('Full Detector: Maximum Amplitude by Channel', fontsize=16)

    fig.colorbar(hist1[3], ax=ax, location='bottom')
    # Customize the plot
    ax.set_xlabel('Channel ID', fontsize=14)
    ax.set_ylabel('Max. Amplitude [ADC]', fontsize=14)
    #output.attach_note('Zero-Indexed Minimal Response Channels:', str(zero_y_bins), positionRect=[0, 0, 0, 0])
    txt = 'Zero-Indexed Minimal Response Channels:'+str(zero_y_bins)
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    plt.legend(loc='upper right', fontsize=14)
    output.savefig()
    plt.close()
##
##
def hist_avgAmp(inpuT, tick_start, tick_end, mod):
    input_avg = np.abs(inpuT[:,:,tick_start:tick_end]).sum(axis=2)/((tick_end-tick_start)*BIT)
    flipped_avg = np.transpose(input_avg)
    return flipped_avg
##
##
def avg_amp(wvfm_matrix, tick_start, tick_end, mod, output, MODULES):

    fig, ax = plt.subplots(figsize=(16, 8))
    
    color_array = ['yellow', 'orange', 'red', 'magenta']
    
    if MODULES==1:
        notflat_avg = flow2sim(wvfm_matrix, hist_avgAmp, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_avg)[-1])
        flat_avg = np.concatenate(notflat_avg, axis=0)
        flat_channel = np.arange(0,96,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, color_array[mod], 0.2, 'Mod '+str(mod))]
        for i in range(0, 96, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
            
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        
        x_range = (0, 96)
        y_range = (0, np.max(flat_avg)*1.1)
        hist1 = ax.hist2d(flat_channels, flat_avg, bins=(96,100), range=[x_range, y_range], norm=mpl.colors.LogNorm(), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none',
                         alpha=0.7, lw=1, label='Dead: '+str(zero_y_bins[i])))
        ax.set_title('Single Module: Average Amplitude by Channel', fontsize=16)

    else: 
        notflat_avg = flow2sim(wvfm_matrix, hist_avgAmp, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_avg)[-1])
        flat_avg = np.concatenate(notflat_avg, axis=0)
        flat_channel = np.arange(0,384,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)

        regions = [(0, 96, 'yellow', 0.2, 'Mod 0'),
                   (96, 192, 'orange', 0.3, 'Mod 1'),
                   (192, 288, 'red', 0.2, 'Mod 2'),
                   (288, 384, 'magenta', 0.2, 'Mod 3'),]
        for i in range(0, 384, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
        
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        ax.axvline(x=144, color='black', linestyle=':')
        ax.axvline(x=192, color='black', linestyle=':')
        ax.axvline(x=240, color='black', linestyle=':')
        ax.axvline(x=288, color='black', linestyle=':')
        ax.axvline(x=336, color='black', linestyle=':')
        ax.axvline(x=384, color='black', linestyle=':')
        
        x_range = (0, 384)
        y_range = (0, np.max(flat_avg)*1.1)
        
        hist1 = ax.hist2d(flat_channels, flat_avg, range=[x_range, y_range], bins=(384,100), norm=mpl.colors.LogNorm(vmax=50), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none',
                         alpha=0.7, lw=1))        
        
        ax.set_title('Full Module: Average Amplitude by Channel', fontsize=16)
    #output.attach_note('Zero-Indexed Minimal Response Channels:', str(zero_y_bins), positionRect=[0, 0, 0, 0])
    txt = 'Zero-Indexed Minimal Response Channels:'+str(zero_y_bins)
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    fig.colorbar(hist1[3], ax=ax, location='bottom')
    # Customize the plot
    ax.set_xlabel('Channel ID', fontsize=14)
    ax.set_ylabel('Average |Amplitude| [ADC]', fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    output.savefig()
    plt.close()
##
##
def hist_ampRatio(inpuT, tick_start, tick_end, mod):
    denominator_mask = (np.mean(np.abs(inpuT[:,:,0:tick_start])/BIT, axis=2) == 0)
    numerator = np.mean(np.abs(inpuT[:,:,tick_end:-1])/BIT, axis=2)*(denominator_mask==0)
    denominator = np.where(denominator_mask == 1, 1, np.mean(np.abs(inpuT[:,:,0:tick_start])/BIT, axis=2))
    
    input_ratio = numerator / denominator
    
    flipped_ratio = np.transpose(input_ratio)
    return flipped_ratio
##
##
def amp_ratio(wvfm_matrix, tick_start, tick_end, mod, output, MODULES):

    fig, ax = plt.subplots(figsize=(16, 8)) 
    color_array = ['yellow', 'orange', 'red', 'magenta']
    
    if MODULES==1: 
        notflat_ratio = flow2sim(wvfm_matrix, hist_ampRatio, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_ratio)[-1])
        flat_ratio = np.concatenate(notflat_ratio, axis=0)
        flat_channel = np.arange(0,96,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, color_array[mod], 0.2, 'Mod '+str(mod))]
        for i in range(0, 96, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
            
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        
        x_range = (0, 96)
        y_range = (0, np.max(flat_ratio)*1.1)
        hist1 = ax.hist2d(flat_channels, flat_ratio, bins=(96,100), range=[x_range, y_range], norm=mpl.colors.LogNorm(), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none', alpha=0.7, lw=1))
        ax.set_title('Single Module: Ratios of the Average Noise Amplitude [800:1000]/[0:200]', fontsize=16)
        
    else: 
        notflat_ratio = flow2sim(wvfm_matrix, hist_ampRatio, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_ratio)[-1])
        flat_ratio = np.concatenate(notflat_ratio, axis=0)
        flat_channel = np.arange(0,384,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, 'yellow', 0.2, 'Mod 0'),
                   (96, 192, 'orange', 0.3, 'Mod 1'),
                   (192, 288, 'red', 0.2, 'Mod 2'),
                   (288, 384, 'magenta', 0.2, 'Mod 3'),]
        for i in range(0, 384, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
        
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        ax.axvline(x=144, color='black', linestyle=':')
        ax.axvline(x=192, color='black', linestyle=':')
        ax.axvline(x=240, color='black', linestyle=':')
        ax.axvline(x=288, color='black', linestyle=':')
        ax.axvline(x=336, color='black', linestyle=':')
        ax.axvline(x=384, color='black', linestyle=':')
        
        x_range = (0, 384)
        y_range = (0, 20)
        
        hist1 = ax.hist2d(flat_channels, flat_ratio, bins=(384,200), range=[x_range, y_range], norm=mpl.colors.LogNorm(vmax=1e2), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        for i in range(len(zero_y_bins)):
            ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
                         zero_y_bins[i]+1 - zero_y_bins[i], 
                         y_range[1] - y_range[0], edgecolor='red', facecolor='none', alpha=0.7, lw=1))
        ax.set_title('Full Detector: Ratios of the Average Noise Amplitude [800:1000]/[0:200]', fontsize=16)

    txt = 'Zero-Indexed Minimal Response Channels:'+str(zero_y_bins)
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    fig.colorbar(hist1[3], ax=ax, location='bottom')
    # Customize the plot
    ax.set_xlabel('Channel ID', fontsize=14)
    ax.set_ylabel('Ratio End/Pretrigger', fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    txt = ' '
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    output.savefig()
    plt.close()
##  
## Dark Count Plot Functions ##
##
def hist_darkR8(inpuT, tick_start, tick_end, mod):
    kernel = np.array([1, 2, 1])
    y_smooth = np.apply_along_axis(lambda m: convolve(m, kernel, mode='valid'), axis=-1, arr=inpuT[:,:,tick_start:tick_end]/BIT)/4
        
    def find_peaks_safe(arr):
        peaks, _ = find_peaks(arr,height=200)
        return len(peaks) if peaks.size > 0 else 0

    peaks = np.apply_along_axis(lambda m: find_peaks_safe(m), axis=-1, arr=np.abs(y_smooth))
    flipped_peaks = np.transpose(peaks)
    return flipped_peaks
##
##
def dark_count(wvfm_matrix, tick_start, tick_end, mod, output, MODULES):

    fig, ax = plt.subplots(figsize=(16, 8))
    color_array = ['yellow', 'orange', 'red', 'magenta']
    
    if MODULES==1:
        notflat_ratio = flow2sim(wvfm_matrix, hist_darkR8, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_ratio)[-1])
        flat_ratio = np.concatenate(notflat_ratio, axis=0)
        flat_channel = np.arange(0,96,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, color_array[mod], 0.2, 'Mod '+str(mod))]
        for i in range(0, 96, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
            
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        
        
        x_range = (0, 96)
        y_range = (0, np.mean(flat_ratio)+40)
        hist1 = ax.hist2d(flat_channels, flat_ratio, bins=(96,np.mean(flat_ratio, dtype=int)+40), range=[x_range, y_range], norm=mpl.colors.LogNorm(vmax=1e3), cmap='viridis')
        zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        #for i in range(len(zero_y_bins)):
        #    ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
        #                 zero_y_bins[i]+1 - zero_y_bins[i], 
        #                 y_range[1] - y_range[0], edgecolor='red', facecolor='none',
        #                 alpha=0.7, lw=1))        
        ax.set_title('Single Module: Dark Count Rate Per Channel', fontsize=16)
    else: 
        notflat_ratio = flow2sim(wvfm_matrix, hist_darkR8, tick_start, tick_end, mod)
        hist_shape = np.int64(np.shape(notflat_ratio)[-1])
        flat_ratio = np.concatenate(notflat_ratio, axis=0)
        flat_channel = np.arange(0,384,1)
        flat_channels = np.concatenate(np.array([flat_channel]*hist_shape).transpose(), axis=0)
        
        regions = [(0, 96, 'yellow', 0.2, 'Mod 0'),
                   (96, 192, 'orange', 0.3, 'Mod 1'),
                   (192, 288, 'red', 0.2, 'Mod 2'),
                   (288, 384, 'magenta', 0.2, 'Mod 3'),]
        for i in range(0, 384, 12):
            ax.axvspan(i+6, i + 12, alpha=0.2, color='green')
        
        for xmin, xmax, color, alpha, label in regions:
            ax.axvspan(xmin, xmax, facecolor=color, alpha=alpha, label=label)
        #ax.axhline(y=1, color='red', linestyle='--', label='Min = 1')
        ax.axvline(x=48, color='black', linestyle=':')
        ax.axvline(x=96, color='black', linestyle=':')
        ax.axvline(x=144, color='black', linestyle=':')
        ax.axvline(x=192, color='black', linestyle=':')
        ax.axvline(x=240, color='black', linestyle=':')
        ax.axvline(x=288, color='black', linestyle=':')
        ax.axvline(x=336, color='black', linestyle=':')
        ax.axvline(x=384, color='black', linestyle=':')
        
        x_range = (0, 384)
        y_range = (0, np.mean(flat_ratio)+40)
        
        hist1 = ax.hist2d(flat_channels, flat_ratio, bins=(384,np.mean(flat_ratio, dtype=int)+40), range=[x_range, y_range], norm=mpl.colors.LogNorm(vmax=7e2), cmap='viridis')
        #zero_y_bins = np.where(hist1[0][:,1:].sum(axis=-1) == 0)[0]
        #for i in range(len(zero_y_bins)):
        #    ax.add_patch(plt.Rectangle((zero_y_bins[i], 0), 
        #                 zero_y_bins[i]+1 - zero_y_bins[i], 
        #                 y_range[1] - y_range[0], edgecolor='red', facecolor='none',
        #                 alpha=0.7, lw=1))        
        ax.set_title('Full Detector: Dark Count Rate Per Channel', fontsize=16)

    fig.colorbar(hist1[3], ax=ax, location='bottom')
    # Customize the plot
    ax.set_xlabel('Channel ID', fontsize=14)
    ax.set_ylabel('Dark Counts Per Wvfm', fontsize=14)
    #output.attach_note('Zero-Indexed Minimal Response Channels:', str(zero_y_bins), positionRect=[0, 0, 0, 0])
    #txt = 'Zero-Indexed Minimal Response Channels:'+str(zero_y_bins)
    #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16)
    plt.legend(loc='upper left', fontsize=14)
    txt = ' '
    plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    output.savefig()
    plt.close()
##
## End Plotting Functions ##

def main(input_file, dead_json, output_file_1, output_file_2, output_file_3):

    ## open json ##
    with open(dead_json, 'r') as file:
        data = json.load(file)

    # Extract arrays from the JSON data
    dead_channels = data.get('dead_channels')
    inactive_channels = data.get('inactive_channels')
    bad_baseline = data.get('bad_baseline')
    
    ## open file ##
    file = h5py.File(input_file, 'r')

    ## define the light waveform matrix ##
    wvfm = file["light/wvfm/data"][::3]#['samples']#[::10,:,:,:]
    #del file
    #light_wvfm_start = wvfm['samples']
    ## mask out inactive channels and remove pedestals##
    light_wvfms_ped = wvfm['samples'][:,:,sipm_channels,:]
    #del light_wvfm_start
    light_wvfms = light_wvfms_ped.astype(float) - light_wvfms_ped[:,:,:50].mean(axis=-1, keepdims=True)
    del light_wvfms_ped
    ## define the number of ADCs in data ##
    MODULES = int(np.shape(light_wvfms)[1]/2)
    #SAMPLES = np.shape(light_wvfms)[-1]
    ## Livio's Plots: ##
    #wvfm = file["light/wvfm/data"]['samples']
    wvfm_alL = np.zeros((wvfm.shape[0],wvfm['samples'].shape[1],wvfm['samples'].shape[2],wvfm['samples'].shape[3]))
    for i in range(wvfm.shape[0]):
        wvfm_alL[i,:,:,:] = wvfm[i][0]
    wvfm_all = wvfm_alL.astype(float) - wvfm_alL[:,:,:,:50].mean(axis=-1, keepdims=True)
    print('wvfm_all shape',np.shape(wvfm_all))
    del wvfm

    with PdfPages(output_file_1) as output1:
        output2 = output_file_2
        output3 = output_file_3
        
        # First Plots: Check baseline average per channel in one file:
        
        try:
            bsline_pFile(wvfm_alL, MOD, output1, output2, MODULES, data)
            print('1/12')
        except: 
            txt = 'Error: Baseline Plot Averaged Over File' 
            print(txt)
            #plt.text(0.05,0.95, txt, transform=fig.transFigure, size=16)
            
        # Check if the Baseline Drifts across the file:
        try:
            bsline_drift(wvfm_alL, MOD, output1, MODULES)
            print('2/12')
        except:
            txt = 'Error: Baseline Drift Plot Over File' 
            print(txt)
            #plt.text(0.05,0.95, txt, transform=fig.transFigure, size=16)
        
        # Checking trigger lineup:
        #try:
        #    signal_region(light_wvfms, output1, 3)
        print('Skipping 3/12')
        #except: 
        #    txt = 'Error: Average Beam Alignment Plot Over File' 
        #    print(txt)
            #plt.text(0.05,0.95, txt, transform=fig.transFigure, size=16)      
        
        # The third variable, skip int, determines the gap between plotted wvfms
        try:
            trigger_timing(light_wvfms, range(0,light_wvfms.shape[0],10), output1)
            print('4/12')
        except: 
            txt = 'Error: Beam Alignment Plot Over File' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
        # Second Plot: Check the Max. Amplitude at each tick, averaged for all active channels on an ADC and events in file
        #try:
        #    if MODULES==1:
        #        LCM_wvfm = np.transpose(light_wvfms[:,1,:,:], (1,0,2))
        #        ACL_wvfm = np.transpose(light_wvfms[:,0,:,:], (1,0,2))
        #        amp_by_tick(ACL_wvfm, 0, 0, 1000, 'ACL', 0, 'Purples', output1)
        #        amp_by_tick(LCM_wvfm, 0, 0, 1000, 'LCM', 0, 'Greens', output1)
    
        #    else: 
        #        for i in range(MODULES):
        #            LCM_wvfm = np.transpose(light_wvfms[:,(i*2)+1,:,:], (1,0,2))
        #            ACL_wvfm = np.transpose(light_wvfms[:,(i*2),:,:], (1,0,2))
        #            amp_by_tick(ACL_wvfm, 0, 0, 1000, 'ACL', i, 'Purples', output1)
        #            amp_by_tick(LCM_wvfm, 0, 0, 1000, 'LCM', i, 'Greens', output1)
        print('Skipping 5/12')
        #except: 
        #    txt = 'Error: Average Max. Amplitude Per Tick Plot' 
        #    print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16)
            
        # Third Plot: Check the Max. Amplitude at each event in a file, averaged for all active channels on an ADC and ticks in wvfm
        try: 
            maxAmp_pModpEv(wvfm_all, MOD, output1, MODULES)
            print('6/12')
        except: 
            txt = 'Error: Max. Amplitude per Event Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
        # Fourth Plot: Check the Max. Amplitude at each channel, averaged across all events in a file and ticks in wvfm
        try: 
            maxAmp_pChanpFile(wvfm_all, MOD, output1, output3, MODULES, data)
            print('7/12')
        except: 
            txt = 'Error: Max. Amplitude per Channel Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16)
        
        # Fifth Plot: Check the Max. Amplitude at each channel for each event in the file (histogram):
        try: 
            max_amp(light_wvfms, 0, 1000, MOD, output1, MODULES)
            print('8/12')
        except: 
            txt = 'Error: Another Max. Amplitude Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
        # Sixth Plot: Check the Mean. Amplitude across all ticks for each channel, for each event in file:
        try:
            avg_amp(light_wvfms, 0, 1000, MOD, output1, MODULES)
            print('9/12')
        except: 
            txt = 'Error: Mean Amplitude Across All Ticks Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
        # Seventh Plot: Check the ratio of mean amplitude for early vs late ticks, for each channel, for each event:
        try:
            amp_ratio(light_wvfms, 200, 800, MOD, output1, MODULES)
            print('10/12')
        except: 
            txt = 'Error: Mean Amplitude Late vs. Early Ticks Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
        # Eighth Plot: Check the dark count rate across a waveform for each channel, for each event:
        try:
            dark_count(light_wvfms, 0, 1000, MOD, output1, MODULES)
            print('11/12')
        except:
            txt = 'Error: Dark Count Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
            
        # Ninth Plot: Check the FFTs for each ADC
        try: 
            short_l_wvfms = light_wvfms[::2,:,:,:]
            length = np.shape(short_l_wvfms)[0]
            if length > 500:
                max_cap = 500
            else: 
                max_cap = length
            
            if MODULES==1:
                ACL_dataset = noise_datasets(ak.flatten(short_l_wvfms[:max_cap,0,:,:], axis=1), THRESHOLD)
                LCM_dataset = noise_datasets(ak.flatten(short_l_wvfms[:max_cap,1,:,:], axis=1), THRESHOLD)
                ACL_maxes = power_hist_maxes(ACL_dataset)
                LCM_maxes = power_hist_maxes(LCM_dataset)
    
                power_spec_plots(ACL_dataset, ACL_maxes, LCM_dataset, LCM_maxes, PRE_NOISE, 0, output1)
            else: 
                for i in range(MODULES):
                    ACL_dataset = noise_datasets(ak.flatten(short_l_wvfms[:max_cap,(i*2),:,:], axis=1), THRESHOLD)
                    LCM_dataset = noise_datasets(ak.flatten(short_l_wvfms[:max_cap,(i*2)+1,:,:], axis=1), THRESHOLD)
                    ACL_maxes = power_hist_maxes(ACL_dataset)
                    LCM_maxes = power_hist_maxes(LCM_dataset)
        
                    power_spec_plots(ACL_dataset, ACL_maxes, LCM_dataset, LCM_maxes, PRE_NOISE, i, output1)
            print('12/12')
        except:
            txt = 'Error: Noise FFT Plot' 
            print(txt)
            #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=16) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str,help='''string corresponding to the path of the flow_file output data file to be considered''')
    parser.add_argument('--dead_json', default=None, type=str,help='''string corresponding to the path of the json input file with known issues recorded''')
    parser.add_argument('--output_file_1', default=None, type=str, help='Main Output PDF file')
    parser.add_argument('--output_file_2', default=None, type=str, help='Baseline Offset Output PNG file')
    parser.add_argument('--output_file_3', default=None, type=str, help='Dead Channels Output PNG file')
    args = parser.parse_args()
    main(**vars(args))
