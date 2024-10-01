#!/usr/bin/env python3

##########################################
##                                      ##
##          ~ FSD Light DQM ~           ##
##                                      ##
##########################################
##
##  Written 30.09.2024
##
##    - Angela White: ajwhite@uchicago.edu
##
##  Version:  v01, 30.09.2024, direct questions to A. J. White
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

sipm_channels = ([0,1,2,3,4,5] + \
                 [6,7,8,9,10,11] + \
                 [12,13,14,15,16,17] + \
                 [18,19,20,21,22,23] + \
                 [24,25,26,27,28,29] + \
                 [32,33,34,35,36,37] + \
                 [38,39,40,41,42,43] + \
                 [44,45,46,47,48,49] + \
                 [50,51,52,53,54,55] + \
                 [56,57,58,59,60,61])

## Plotting Functions ##

## Function for Baseline ##
## 
def emoji_thang(bsln_arr, input1, input2, adc):
    
    length = 64
    expected = np.ones(length)
    inactive = input1
    special_value = 0
    expected[inactive] = special_value
    bbase = input2[f'{adc}']
    expected[bbase] = special_value
    expected = np.int64(expected)
    baseline = bsln_arr[adc]
    baseline[inactive] = special_value
    baseline[bbase] = special_value

    if np.array_equal(baseline, expected) == True:
        if np.sum(baseline, dtype=int) == 60:
            color = 'limegreen'
            text = emoji.emojize(':smile:',language='alias')
        else:
            color = 'deeppink'
            text = emoji.emojize(':winking_face:',language='alias')           
    else:
        color = 'orangered'
        text = emoji.emojize(':angry:',language='alias')
    
    return color, text
##
##
def baseline_graf(bsln_arr, mod, input1, input2, output):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # get the arrangement right for A, B, +z, -z:
    ordering_row = [1, 0, 1, 0]
    ordering_col = [1, 1 ,0, 0]
    ordering_adc = [0, 1, 2, 3]

    fig, axs = plt.subplots(2, 2, figsize=(4, 4))
    for k in range(4):
        i = ordering_row[k]
        j = ordering_col[k]
        l = ordering_adc[k]
        color, text = emoji_thang(bsln_arr, input1, input2, l)
        # Plot rectangle
        axs[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        axs[i, j].set_xticks([])  # Remove x-axis ticks
        axs[i, j].set_yticks([])  # Remove y-axis ticks
        axs[i, j].tick_params(axis='both', which='both', length=0)  # Hide tick marks
        axs[i, j].annotate('{}'.format(current_time), xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)
        axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

    # Column headers
    col_headers = ["TPC B", "TPC A"]
    for ax, col_header in zip(axs[0], col_headers):
        ax.annotate(col_header, xy=(0.5, 1.1), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

    # Row headers
    row_headers = ["+z", "-z"]
    for ax, row_header in zip(axs[:, 0], row_headers):
        ax.annotate(row_header, xy=(-0.2, 0.5), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output)
    plt.close()
#
#
def bsline_pFile(wvfm_all, mod, output1, output2, MODULES, inputs):
    
    max_amp_pchan = []
    
    baseline_list = []
    
    for mod in range(4):
        max_amp_pchan.append(wvfm_all[:,mod,:,:50].mean(axis=-1).mean(axis=0))

    fig, ax = plt.subplots(4,figsize=(12,8))
    for mod in range(4):
        ax[mod].plot(max_amp_pchan[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[mod].set_ylabel("Baseline")
            
        baseline_list.append(max_amp_pchan[mod])
    
    for iax in ax.reshape(-1):
        iax.xaxis.set_major_locator(MultipleLocator(63))
        iax.xaxis.set_minor_locator(FixedLocator([29.5,31.5,61.5]))
        iax.grid(axis = 'x',which="both")
        #iax.set_ylim(-1000,sat+1000)

    row_headers=["ADC 0", "ADC 1", "ADC 2", "ADC 3"]
    for axs, row_header in zip(ax[:], row_headers):
        axs.annotate(row_header, xy=(1.01, 0.3), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold', rotation=270)

    plt.tight_layout()
    plt.show()
    output1.savefig()
    plt.close()
        
    baseline_array = np.array(baseline_list)
    baseline_mask = (baseline_array > -2300) & (baseline_array < 2300)
    baseline_test = np.int64(baseline_array*(baseline_mask==0)+baseline_mask)
    baseline_test[baseline_test != 1] = 0
    baseline_graf(baseline_test, mod, inputs['inactive_channels'], inputs['bad_baseline'], output2)
#
#
def maxamp_graf(mxamp_arr, mod, input1, input2, output):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Loop through each subplot
    fig, axs = plt.subplots(2, 2, figsize=(4, 4))
    # get the arrangement right for A, B, +z, -z:
    ordering_row = [1, 0, 1, 0]
    ordering_col = [1, 1 ,0, 0]
    ordering_adc = [0, 1, 2, 3]

    for k in range(4):
        i = ordering_row[k]
        j = ordering_col[k]
        l = ordering_adc[k]
        color, text = emoji_thang(mxamp_arr, input1, input2, l)
        # Plot rectangle
        axs[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].tick_params(axis='both', which='both', length=0)  # Hide tick marks
        axs[i, j].annotate('{}'.format(current_time), xy=(0.5, 0.05), xycoords='axes fraction', ha='center', fontsize=8)
        axs[i, j].annotate('{}'.format(text), xy=(0.5, 0.3), xycoords='axes fraction', ha='center', fontsize=60)

    # Column headers
    col_headers = ["TPC B", "TPC A"]
    for ax, col_header in zip(axs[0], col_headers):
        ax.annotate(col_header, xy=(0.5, 1.1), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

    # Row headers
    row_headers = ["+z", "-z"]
    for ax, row_header in zip(axs[:, 0], row_headers):
        ax.annotate(row_header, xy=(-0.2, 0.5), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold')

    #plt.title("LRS Dead Channels")        
    plt.savefig(output)
    plt.close()
#
#
def maxAmp_pChanpFile(wvfm_all, mod, output1, output3, MODULES, inputs):
    
    max_amp_pchan = []
    deadchan_list = []
    
    for mod in range(4):
        max_amp_pchan.append(wvfm_all[:,mod,:,:].max(axis=0).max(axis=-1))

    fig, ax = plt.subplots(4,figsize=(12,8))
    for mod in range(4):
        ax[mod].plot(max_amp_pchan[mod],marker=".",markerfacecolor='black',markeredgecolor='None',linestyle='None')
        ax[mod].set_ylabel("Max Amp [ADC]")
            
        deadchan_list.append(max_amp_pchan[mod])
    
    for iax in ax.reshape(-1):
        iax.xaxis.set_major_locator(MultipleLocator(63))
        iax.xaxis.set_minor_locator(FixedLocator([29.5,31.5,61.5]))
        iax.grid(axis = 'x',which="both")
        iax.set_yscale('log')
        #iax.set_ylim(10,sat*10)
        #iax.axhline(y = sat, color = 'r', linestyle = '-', alpha=0.5)
        #iax.axhline(y = 1000, color = 'r', linestyle = '-', alpha=0.5)

    row_headers=["ADC 0", "ADC 1", "ADC 2", "ADC 3"]
    for axs, row_header in zip(ax[:], row_headers):
        axs.annotate(row_header, xy=(1.01, 0.3), xycoords='axes fraction', ha='center', fontsize=14, fontweight='bold', rotation=270)

    ax[3].set_xlabel("Channel",fontfamily="monospace")
    ax[3].set_xlabel("Channel",fontfamily="monospace")
    #txt = ' '
    #plt.text(0.05,0.95,txt, transform=fig.transFigure, size=12)
    output1.savefig()
    plt.close()
        
    deadchan_array = np.array(deadchan_list)
    deadchan_mask = (deadchan_array > 100)
    deadchan_test = np.int64(deadchan_mask)
    maxamp_graf(deadchan_test, mod, inputs['inactive_channels'], inputs['dead_channels'], output3)  
#
#
def trigger_timing(light_wvfms, events, output):
    
    adc_list = np.arange(np.shape(light_wvfms)[1])
    
    for adc in adc_list:

        transposed_wvfms = np.transpose(light_wvfms, (1, 2, 0, 3))[adc,:,:,:]
    
        fig, ax = plt.subplots(5, 2, figsize=(11.5, 9.5), sharex=True, layout='constrained')
        for summ in range(10):
            start = summ*6
            end = start+6
            flattened_wvfms = transposed_wvfms[start:end, :, :]
            sum_wvfm = np.clip(np.sum(flattened_wvfms, axis=0),None,sat)

            chan = np.shape(flattened_wvfms)[0]
            for ev in events:
                ax[summ//2, summ%2].plot(np.linspace(0,SAMPLES,SAMPLES), sum_wvfm[ev, :], color='grey', linewidth=0.8, label='Sum')
                for i in range(chan):
                    #print('p')
                    ax[summ//2, summ%2].plot(np.linspace(0,SAMPLES,SAMPLES), flattened_wvfms[i,ev,:], linewidth=0.8)
            ax[summ//2, summ%2].set_title('ADC '+str(adc)+': Channels ['+str(start)+':'+str(end-1)+']')
            ax[summ//2, summ%2].set_ylim(-100,33000)
            ax[summ//2, summ%2].grid(True)
            ax[summ//2, summ%2].set_ylim(0, 35000)
            ax[summ//2, 0].set_ylabel('ADC Value')
            ax[4, summ%2].set_xlabel('Sample [0.016 μs]')
            #print('q')

        plt.grid(True)
        rect = plt.Rectangle((0, 0), 1200, 1000, fill=False, edgecolor='black', lw=2)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        plt.legend(unique.values(), unique.keys())
        fig.patches.append(rect)
        output.savefig()
        plt.close()
        #plt.show()
#
#
def uniform_fft(light_wvfms, output):
    
    adc_list = np.arange(np.shape(light_wvfms)[1])
    freq = np.fft.rfftfreq(SAMPLES, d=SAMPLE_RATE)
    
    for adc in adc_list:

        transposed_wvfms = np.transpose(light_wvfms, (1, 2, 0, 3))[adc,sipm_channels,:,:]
    
        fig, ax = plt.subplots(5, 2, figsize=(11.5, 9.5), sharex=True, sharey=True, layout='constrained')
        for summ in range(10):
            start = summ*6
            end = start+6
            flattened_wvfms = transposed_wvfms[start:end, :, :]
            for i in range(6):
                spectra_array = np.zeros((6, 500))
                valid_chan_wvfm = flattened_wvfms[i,:,:]/BIT
        
                # choose first 45 samples as signal-free
                # calculate mean and standard deviation to define signal vs. no sigal
                stdev = np.std(valid_chan_wvfm[:,0:PRE_NOISE], axis=-1)
                mean = np.mean(valid_chan_wvfm[:,0:PRE_NOISE],axis=-1)
                thd = (stdev*3.5) + mean
                no_signal_mask = (valid_chan_wvfm.max(axis=1) < thd)
                # remove the pedestal
                valid_chan_nped = (valid_chan_wvfm.astype(float) - (valid_chan_wvfm[:,0:PRE_NOISE]).mean(axis=-1, keepdims=True))
                noise_wvfms = valid_chan_nped[no_signal_mask==1][:,:SAMPLES]
                
                try:
                    spectrum = np.fft.rfft(noise_wvfms)
                    # remove the DC component
                    spectrum[:,0] = 0
                    normalized_spectrum = np.abs(spectrum[:,:SAMPLES//2]) 
                    # calculate an average fft
                    spectrum_average = np.sum(normalized_spectrum, axis=0) 
                    spectra_array[i] = spectrum_average
                except:
                    #print('pass ADC: '+str(adc)+', Chan: '+str(i))
                    pass
            

                for i in range(6):
                    ax[summ//2, summ%2].plot(freq[1:], spectra_array[i], linewidth=0.8)
                    ax[summ//2, summ%2].set_title('ADC '+str(adc)+': Channels ['+str(start)+':'+str(end-1)+']')
                    ax[summ//2, summ%2].grid(True)
                    ax[summ//2, 0].set_ylabel('Noise Amplitude')
                    ax[4, summ%2].set_xlabel('Freq. [MHz]')

        rect = plt.Rectangle((0, 0), 1200, 1000, fill=False, edgecolor='black', lw=2)
        fig.patches.append(rect)
        output.savefig()
        plt.close()
#
#
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
    wvfm = file["light/wvfm/data"][::10]
    ## mask out inactive channels and remove pedestals##
    light_wvfms_ped = wvfm['samples'][:,:,sipm_channels,:]
    #del light_wvfm_start
    light_wvfms = light_wvfms_ped.astype(float) - light_wvfms_ped[:,:,:PRE_NOISE].mean(axis=-1, keepdims=True)
    del light_wvfms_ped
    ## define the number of ADCs in data ##
    MODULES = int(np.shape(light_wvfms)[1]/4)
    ## Livio's Plots: ##
    #wvfm = file["light/wvfm/data"]['samples']
    wvfm_alL = np.zeros((wvfm.shape[0],wvfm['samples'].shape[1],wvfm['samples'].shape[2],wvfm['samples'].shape[3]))
    for i in range(wvfm.shape[0]):
        wvfm_alL[i,:,:,:] = wvfm[i][0]
    wvfm_all = wvfm_alL.astype(float) - wvfm_alL[:,:,:,:PRE_NOISE].mean(axis=-1, keepdims=True)
    del wvfm

    with PdfPages(output_file_1) as output1:
        output2 = output_file_2
        output3 = output_file_3
        
        # First Plots: Check baseline average per channel in one file:
        
        try:
            bsline_pFile(wvfm_alL, MOD, output1, output2, MODULES, data)
            print('1/4')
        except: 
            txt = 'Error: Baseline Plot Averaged Over File' 
            print(txt)
        
        # The third variable, skip int, determines the gap between plotted wvfms
        try:
            trigger_timing(light_wvfms, range(0,light_wvfms.shape[0],10), output1)
            print('2/4')
        except: 
            txt = 'Error: Beam Alignment Plot Over File' 
            print(txt)
    
        # Fourth Plot: Check the Max. Amplitude at each channel, averaged across all events in a file and ticks in wvfm
        try: 
            maxAmp_pChanpFile(wvfm_all, MOD, output1, output3, MODULES, data)
            print('3/4')
        except: 
            txt = 'Error: Max. Amplitude per Channel Plot' 
            print(txt)
            
        try:
            uniform_fft(wvfm_alL, output1)
            print('4/4')
        except: 
            txt = 'Error: Avg. Noise per Channel Plot' 
            print(txt)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str,help='''string corresponding to the path of the flow_file output data file to be considered''')
    parser.add_argument('--dead_json', default=None, type=str,help='''string corresponding to the path of the json input file with known issues recorded''')
    parser.add_argument('--output_file_1', default=None, type=str, help='Main Output PDF file')
    parser.add_argument('--output_file_2', default=None, type=str, help='Baseline Offset Output PNG file')
    parser.add_argument('--output_file_3', default=None, type=str, help='Dead Channels Output PNG file')
    args = parser.parse_args()
    main(**vars(args))
