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
##  Version: v01, 23.02.2024
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
from scipy.fft import rfft, rfftfreq
import awkward as ak
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.signal import convolve
from scipy.signal import find_peaks
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes


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
PRE_NOISE = 1000 # Will need to be re-defined once we know beam timing
THRESHOLD = 0 # For the FFT, used to define events with signal peaks
## If single module data: could use this for labeling ##
MOD = 0
## Some sort of saturation value re:Livio ##
sat = 32767
    
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
def bsline_pFile(wvfm_all, mod, output, MODULES):
    
    max_amp_pchan_acl = []
    max_amp_pchan_lcm = []
    
    if MODULES==1:
        
        max_amp_pchan_acl.append(wvfm_all[:,0,:,:50].mean(axis=-1).std(axis=0))
        max_amp_pchan_lcm.append(wvfm_all[:,1,:,:50].mean(axis=-1).std(axis=0))

        #fig, ax = plt.subplots(1,2,figsize=(8, 2))
        fig, ax = plt.subplots(1,2,figsize=(16, 4))
        ax[0].plot(max_amp_pchan_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pchan_lcm[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Waveform Baseline [ADC]")
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(16))
            iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
            iax.grid(axis = 'x',which="both")
            iax.set_ylim(-1000,sat+1000)

        row_headers=[f"Mod {mod}"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[0].set_xlabel("ACL")
        ax[1].set_xlabel("LCM") 
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pchan_acl.append(wvfm_all[:,mod*2,:,:50].mean(axis=-1).std(axis=0))
            max_amp_pchan_lcm.append(wvfm_all[:,mod*2+1,:,:50].mean(axis=-1).std(axis=0))

        fig, ax = plt.subplots(4,2,figsize=(16,16))
        for mod in range(4):
            ax[mod,0].plot(max_amp_pchan_acl[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,1].plot(max_amp_pchan_lcm[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,0].set_ylabel("Baseline")
    
        for iax in ax.reshape(-1):
            iax.xaxis.set_major_locator(MultipleLocator(16))
            iax.xaxis.set_minor_locator(FixedLocator([4,10,20,26,36,42,52,58]))
            iax.grid(axis = 'x',which="both")
            iax.set_ylim(-1000,sat+1000)

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("ACL")
        ax[3,1].set_xlabel("LCM")
        txt = ' .'
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output.savefig()
        plt.close()
##
## Functions for FFTs: ##
##
def noise_datasets(no_ped_adc, THRESHOLD):

    adc_signal_indices=[]
    for i in range(len(no_ped_adc)):
        if max(no_ped_adc[i])>THRESHOLD:
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

    ns_wvfms = []

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

    return xcenters, max_bins
##
##
def power_spec_plots(adc0_dataset, adc0_max, adc1_dataset, adc1_max, CUTOFF, mod, output): 
    fig = plt.figure(figsize=(16,4))
    x = np.linspace(0,CUTOFF-1,CUTOFF)
    y0 = adc0_dataset[2][300]/BIT
    y1 = adc1_dataset[2][300]/BIT
    plt.plot(x, y0, "-", color='green', label='ACL')
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
    hist1 = ax[0].hist2d(adc0_freq[adc0_pdbfs>-500], adc0_pdbfs[adc0_pdbfs>-500], bins=32, \
                            norm=mpl.colors.LogNorm(), cmap='viridis')
    fig.colorbar(hist1[3], ax=ax, location='bottom')
    ax[0].plot(adc0_max[0],adc0_max[1],'o-k')
    ax[0].set_title('ACL Noise Power Spectrum')
    ax[0].set_ylim(-110,230)
    ax[0].set_xlabel('Frequency [MHz]',fontsize=14)
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
def flow2sim(waveforms, function, start_tick, end_tick, mod=0):
    
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
            iax.xaxis.set_minor_locator(MultipleLocator(10))
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
            ax[mod,0].set_ylabel("Max amp [ADC]",fontfamily="monospace")
    
        for iax in ax.reshape(-1):
            #iax.xaxis.set_major_locator(MultipleLocator(24))
            iax.xaxis.set_minor_locator(MultipleLocator(10))
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
def maxAmp_pChanpFile(wvfm_all, mod, output, MODULES):
    
    max_amp_pchan_acl = []
    max_amp_pchan_lcm = []
    
    if MODULES==1:
        
        max_amp_pchan_acl.append(wvfm_all[:,0,:,:].max(axis=-1).max(axis=0))
        max_amp_pchan_lcm.append(wvfm_all[:,1,:,:].max(axis=-1).max(axis=0))

        fig, ax = plt.subplots(1,2,figsize=(16,4))
        ax[0].plot(max_amp_pchan_acl[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[1].plot(max_amp_pchan_lcm[0],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[0].set_ylabel("Max amp [ADC]")
    
        for iax in ax.reshape(-1):
            #iax.xaxis.set_major_locator(MultipleLocator(24))
            iax.xaxis.set_minor_locator(MultipleLocator(10))
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
        output.savefig()
        plt.close()
        
        
    else: 

        for mod in range(4):
            max_amp_pchan_acl.append(wvfm_all[:,mod*2,:,:].max(axis=0).max(axis=-1))
            max_amp_pchan_lcm.append(wvfm_all[:,mod*2+1,:,:].max(axis=0).max(axis=-1))

        fig, ax = plt.subplots(4,2,figsize=(16,16))
        for mod in range(4):
            ax[mod,0].plot(max_amp_pchan_acl[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,1].plot(max_amp_pchan_lcm[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
            ax[mod,0].set_ylabel("Max amp [ADC]")
    
        for iax in ax.reshape(-1):
            #iax.xaxis.set_major_locator(MultipleLocator(24))
            iax.xaxis.set_minor_locator(MultipleLocator(10))
            iax.grid(axis = 'x',which="both")
            iax.set_yscale('log')
            iax.set_ylim(10,sat*10)
            iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)

        row_headers=["Mod-0", "Mod-1", "Mod-2", "Mod-3"]
        col_headers=["ACL","LCM"]
        font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        ax[3,0].set_xlabel("Channel",fontfamily="monospace")
        ax[3,1].set_xlabel("Channel",fontfamily="monospace")
        txt = ' '
        plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
        output.savefig()
        plt.close()
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
        notflat_max = flow2sim(wvfm_matrix, hist_maxAmp, tick_start, tick_end)
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
        notflat_avg = flow2sim(wvfm_matrix, hist_avgAmp, tick_start, tick_end)
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
        ax.set_title('Single Module: Ratios of the Average Noise Amplitude [800:1000]/[0:200]', fontsize=16)
        
    else: 
        notflat_ratio = flow2sim(wvfm_matrix, hist_ampRatio, tick_start, tick_end)
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
        ax.set_title('Full Detector: Ratios of the Average Noise Amplitude [800:1000]/[0:200]', fontsize=16)

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
        notflat_ratio = flow2sim(wvfm_matrix, hist_darkR8, tick_start, tick_end)
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

def main(input_file, output_file):

    ## open file ##
    file = h5py.File(input_file, 'r')

    ## define the light waveform matrix ##
    light_wvfm_start = file['light/wvfm/data']['samples']
    ## mask out inactive channels ##
    light_wvfms = light_wvfm_start[:,:,sipm_channels,:]
    ## define the number of ADCs in data ##
    MODULES = int(np.shape(light_wvfms)[1]/2)
    ## Livio's Plots: ##
    wvfm = file["light/wvfm/data"]
    wvfm_all = np.zeros((wvfm.shape[0],wvfm['samples'].shape[1],wvfm['samples'].shape[2],wvfm['samples'].shape[3]))
    for i in range(wvfm.shape[0]):
        wvfm_all[i,:,:,:] = wvfm[i][0]

    with PdfPages(output_file) as output:
        
        # First Plot: Check baseline average per channel in one file:
        bsline_pFile(wvfm_all, MOD, output, MODULES)
    
        # Second Plot: Check the Max. Amplitude at each tick, averaged for all active channels on an ADC and events in file
        if MODULES==1:
            LCM_wvfm = np.transpose(light_wvfms[:,1,:,:], (1,0,2))
            ACL_wvfm = np.transpose(light_wvfms[:,0,:,:], (1,0,2))
            amp_by_tick(ACL_wvfm, 0, 0, 1000, 'ACL', 0, 'Purples', output)
            amp_by_tick(LCM_wvfm, 0, 0, 1000, 'LCM', 0, 'Greens', output)
    
        else: 
            for i in range(MODULES):
                LCM_wvfm = np.transpose(light_wvfms[:,(i*2)+1,:,:], (1,0,2))
                ACL_wvfm = np.transpose(light_wvfms[:,(i*2),:,:], (1,0,2))
                amp_by_tick(ACL_wvfm, 0, 0, 1000, 'ACL', i, 'Purples', output)
                amp_by_tick(LCM_wvfm, 0, 0, 1000, 'LCM', i, 'Greens', output)
            
        # Third Plot: Check the Max. Amplitude at each event in a file, averaged for all active channels on an ADC and ticks in wvfm
        maxAmp_pModpEv(wvfm_all, MOD, output, MODULES)
    
        # Fourth Plot: Check the Max. Amplitude at each channel, averaged across all events in a file and ticks in wvfm
        maxAmp_pChanpFile(wvfm_all, MOD, output, MODULES)
    
        # Fifth Plot: Check the Max. Amplitude at each channel for each event in the file (histogram):
        max_amp(light_wvfms, 0, 1000, MOD, output, MODULES)
    
        # Sixth Plot: Check the Mean. Amplitude across all ticks for each channel, for each event in file:
        avg_amp(light_wvfms, 0, 1000, MOD, output, MODULES)
    
        # Seventh Plot: Check the ratio of mean amplitude for early vs late ticks, for each channel, for each event:
        amp_ratio(light_wvfms, 200, 800, MOD, output, MODULES)
    
        # Eighth Plot: Check the dark count rate across a waveform for each channel, for each event:
        dark_count(light_wvfms, 0, 1000, MOD, output, MODULES)
        
        # Ninth Plot: Check the FFTs for each ADC
        length = np.shape(light_wvfm_start[:,:,sipm_channels,:])[0]
        if length > 500:
            max_cap = 500
        else: 
            max_cap = length
            
        if MODULES==1:
            ACL_dataset = noise_datasets(ak.flatten(light_wvfms[:max_cap,0,:,:], axis=1), THRESHOLD)
            LCM_dataset = noise_datasets(ak.flatten(light_wvfms[:max_cap,1,:,:], axis=1), THRESHOLD)
            ACL_maxes = power_hist_maxes(ACL_dataset)
            LCM_maxes = power_hist_maxes(LCM_dataset)
    
            power_spec_plots(ACL_dataset, ACL_maxes, LCM_dataset, LCM_maxes, PRE_NOISE, 0, output)
        else: 
            for i in range(MODULES):
                ACL_dataset = noise_datasets(ak.flatten(-light_wvfms[:max_cap,(i*2),:,:], axis=1), THRESHOLD)
                LCM_dataset = noise_datasets(ak.flatten(light_wvfms[:max_cap,(i*2)+1,:,:], axis=1), THRESHOLD)
                ACL_maxes = power_hist_maxes(ACL_dataset)
                LCM_maxes = power_hist_maxes(LCM_dataset)
        
                power_spec_plots(ACL_dataset, ACL_maxes, LCM_dataset, LCM_maxes, PRE_NOISE, i, output)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str,help='''string corresponding to the path of the flow_file output data file to be considered''')
    parser.add_argument('--output_file', default=None, type=str, help='Output PDF file')
    args = parser.parse_args()
    main(**vars(args))
