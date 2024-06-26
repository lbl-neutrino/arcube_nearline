#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from datetime import datetime
import matplotlib.dates as mdates
import csv
import json
import yaml
from copy import copy
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
import argparse
from collections import defaultdict


centerline_chip_id=list(range(20,120,10))
horizontal_chip_id=list(range(11,21,1))+list(range(101,111,1))

centerline_channel_id=[60,52,53,48,45,41,35]
horizontal_channel_id=[60,61,62,63,1,2,3]+[35,34,33,31,30,29,28]

def _default_pxy():
    return (0., 0.)


def _rotate_pixel(pixel_pos, tile_orientation):
    return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]


def tile_to_io_channel(tile):
    io_channel=[]
    for t in tile:
        for i in range(1,5,1): io_channel.append( ((t-1)*4)+i )
    return io_channel


def load_multitile_geometry(geometry_path):
    
    geo = None
    
    with open(geometry_path) as fi:
        geo = yaml.full_load(fi)

    # Adapted from: https://github.com/larpix/larpix-v2-testing-scripts/blob/master/event-display/evd_lib.py

    pixel_pitch = geo['pixel_pitch']

    chip_channel_to_position = geo['chip_channel_to_position']
    tile_orientations = geo['tile_orientations']
    tile_positions = geo['tile_positions']
    tpc_centers = geo['tpc_centers']
    tile_indeces = geo['tile_indeces']
    xs = np.array(list(chip_channel_to_position.values()))[
        :, 0] * pixel_pitch
    ys = np.array(list(chip_channel_to_position.values()))[
        :, 1] * pixel_pitch
    x_size = max(xs)-min(xs)+pixel_pitch
    y_size = max(ys)-min(ys)+pixel_pitch

    tile_geometry = defaultdict(int)
    io_group_io_channel_to_tile = {}
    geometry = defaultdict(_default_pxy)

    for tile in geo['tile_chip_to_io']:
        tile_orientation = tile_orientations[tile]
        tile_geometry[tile] = tile_positions[tile], tile_orientations[tile]
        for chip in geo['tile_chip_to_io'][tile]:
            io_group_io_channel = geo['tile_chip_to_io'][tile][chip]
            io_group = io_group_io_channel//1000
            io_channel = io_group_io_channel % 1000
            io_group_io_channel_to_tile[(
                io_group, io_channel)] = tile

        for chip_channel in geo['chip_channel_to_position']:
            chip = chip_channel // 1000
            channel = chip_channel % 1000
            try:
                io_group_io_channel = geo['tile_chip_to_io'][tile][chip]
            except KeyError:
                print("Chip %i on tile %i not present in network" %
                      (chip, tile))
                continue

            io_group = io_group_io_channel // 1000
            io_channel = io_group_io_channel % 1000
            x = chip_channel_to_position[chip_channel][0] * \
                pixel_pitch + pixel_pitch / 2 - x_size / 2
            y = chip_channel_to_position[chip_channel][1] * \
                pixel_pitch + pixel_pitch / 2 - y_size / 2

            x, y = _rotate_pixel((x, y), tile_orientation)
            x += tile_positions[tile][2] + \
                tpc_centers[tile_indeces[tile][0]][0]
            y += tile_positions[tile][1] + \
                tpc_centers[tile_indeces[tile][0]][1]

            geometry[(io_group, io_group_io_channel_to_tile[(
                io_group, io_channel)], chip, channel)] = x, y
    
    geometry['pixel_pitch'] = pixel_pitch

    return geometry


def anode_xy(ax, geo, d, metric, normalization, iog):
    
    cmap = matplotlib.colormaps['viridis']
    
    if metric=='mean': metric=0
    if metric=='std': metric=1
     

    if metric == 0: ax.set_title('io_group '+str(iog)+" Mean")
    elif metric == 1: ax.set_title('io_group '+str(iog)+" RMS")
    

    grid_tpc = np.zeros((7*10*4,7*10*2))
    npix = 70
    if iog in [5,6]:
        grid_tpc = np.zeros((8*10*4,8*10*2))
        npix = 80
    
    pitch=geo['pixel_pitch']

    for i in range(1,9,1):
        io_channels=tile_to_io_channel([i])
        tile = i + 8 * (1 - (iog % 2))
        for chipid in range(11, 111):
            for j in [iog]:
                for ioc in io_channels:
                    chip_key=str(j)+'-'+str(ioc)+'-'+str(chipid)
                    if chip_key not in d: continue
                    for channelid in range(64):
                        if channelid in [6,7,8,9,22,23,24,25,38,39,40,54,55,56,57] and iog not in [5,6]: continue
                        if d[chip_key][channelid][0]==-1: continue
                        weight=d[chip_key][channelid][metric]/normalization

                        x, y = geo[(2 - (iog % 2), tile, chipid, channelid)]
                        
                        # print(chip_key, '-', channelid)
                        # print(x, ', ', y)
                        x_ind = int((x + npix*pitch)/pitch) 
                        y_ind = int((y + npix*2*pitch)/pitch) 
                        # print(x_ind, ', ', y_ind)
                        grid_tpc[y_ind][x_ind] = weight
    
    ax.imshow(grid_tpc,interpolation='none',cmap='viridis',vmin=0.,vmax=1.,origin='lower')
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])

    return 1


def parse_sc_data(csv_file):

    data = {}

    with open(csv_file,newline='') as file:
        reader = csv.reader(file,delimiter=';')
        for row in reader:
            if row[1] == 'Time': continue
            timestamp = datetime.fromisoformat(row[1])
            if row[0] in data.keys():
                #data[row[0]].append([timestamp,float(row[2])])
                data[row[0]][0].append(timestamp)
                data[row[0]][1].append(float(row[2]))
            else:
                data[row[0]] = [[timestamp],[float(row[2])]]
            #print(row[0])
            #print(row[1])
            #print(row[2])
    return data

def make_summary_plot(d, ax):

    mu_bulk_1, std_bulk_1, mu_centerline_1, std_centerline_1, mu_horizontal_1, std_horizontal_1 = [[] for i in range(6)]
    mu_bulk_2, std_bulk_2, mu_centerline_2, std_centerline_2, mu_horizontal_2, std_horizontal_2 = [[] for i in range(6)]
    for key in d.keys():
        iog=int(key.split('-')[0])
        ioc=int(key.split('-')[1])
        cid=int(key.split('-')[2])
        if iog==1:
            for channel_id in range(len(d[key])):
                if cid in centerline_chip_id and \
                   channel_id in centerline_channel_id:
                    mu_centerline_1.append(d[key][channel_id][0])
                    std_centerline_1.append(d[key][channel_id][1])
                elif cid in horizontal_chip_id and \
                     channel_id in horizontal_channel_id:
                    mu_horizontal_1.append(d[key][channel_id][0])
                    std_horizontal_1.append(d[key][channel_id][1])
                else:
                    mu_bulk_1.append(d[key][channel_id][0])
                    std_bulk_1.append(d[key][channel_id][1])
        if iog==2:
            for channel_id in range(len(d[key])):
                if cid in centerline_chip_id and \
                   channel_id in centerline_channel_id:
                    mu_centerline_2.append(d[key][channel_id][0])
                    std_centerline_2.append(d[key][channel_id][1])
                elif cid in horizontal_chip_id and \
                     channel_id in horizontal_channel_id:
                    mu_horizontal_2.append(d[key][channel_id][0])
                    std_horizontal_2.append(d[key][channel_id][1])
                else:
                    mu_bulk_2.append(d[key][channel_id][0])
                    std_bulk_2.append(d[key][channel_id][1])
    bins=np.linspace(0,50,51)
    ax[0].hist(mu_bulk_1, bins=bins,label='bulk (TPC 1)', histtype='step', density=True)
    ax[0].hist(mu_centerline_1, bins=bins, label='centerline (TPC 1)', histtype='step', density=True)
    ax[0].hist(mu_horizontal_1, bins=bins, label='horizontal (TPC 1)', histtype='step', density=True)
    ax[0].hist(mu_bulk_2, bins=bins,label='bulk (TPC 2)', histtype='step', density=True)
    ax[0].hist(mu_centerline_2, bins=bins, label='centerline (TPC 2)', histtype='step', density=True)
    ax[0].hist(mu_horizontal_2, bins=bins, label='horizontal (TPC 2)', histtype='step', density=True)
    
    bins=np.linspace(0,10,41)
    ax[1].hist(std_bulk_1, bins=bins,label='bulk (TPC 1)', histtype='step', density=True)
    ax[1].hist(std_centerline_1, bins=bins, label='centerline (TPC 1)', histtype='step', density=True)
    ax[1].hist(std_horizontal_1, bins=bins, label='horizontal (TPC 1)', histtype='step', density=True)
    ax[1].hist(std_bulk_2, bins=bins,label='bulk (TPC 2)', histtype='step', density=True)
    ax[1].hist(std_centerline_2, bins=bins, label='centerline (TPC 2)', histtype='step', density=True)
    ax[1].hist(std_horizontal_2, bins=bins, label='horizontal (TPC 2)', histtype='step', density=True)
    for i in range(2):
        ax[i].grid(True)
        if i==0: ax[i].set_xlabel('ADC Mean')
        if i==1:
            ax[i].set_xlabel('ADC RMS')
            ax[i].legend()

def date_from_ped_filename(filename):
    #date_str = filename.split('-')[-1].split('.json')[0]
    #date_str = filename.split('-')[-1].split('CET.json')[0]
    date_str = filename.split('-')[-1].split('_CDT.h5.panel_ped.json')[0]
    #timestamp = datetime.fromisoformat(date_str)
    print(date_str)
    #timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M%Z")
    timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M_%S")
    return timestamp

def main(input_file, output_file):

    timestamp = date_from_ped_filename(input_file)

    with open(input_file,'r') as f: d = json.load(f)
    geo_mod0 = load_multitile_geometry('layouts_v4/multi_tile_layout-2.3.16_mod0_swap_T8T4T7.yaml')
    geo_mod1 = load_multitile_geometry('layouts_v4/multi_tile_layout-2.3.16_mod1_noswap.yaml')
    geo_mod2 = load_multitile_geometry('layouts_v4/multi_tile_layout-2.5.16_mod2_swap_T7T8.yaml')
    geo_mod3 = load_multitile_geometry('layouts_v4/multi_tile_layout-2.3.16_mod3_swap_T5T8.yaml')

    fig = plt.figure(constrained_layout=True,figsize=(30,16))
    fig_grid = fig.add_gridspec(ncols=9,nrows=2,height_ratios=[1,1],width_ratios=[1,1,1,1,1,1,1,1,0.2])
    anode_mean_plots = []
    anode_rms_plots = []
    fig.suptitle(timestamp,fontsize=18)
    for iog in [1,2,3,4,5,6,7,8]:
        ax_mean = fig.add_subplot(fig_grid[0,iog-1])
        anode_mean_plots.append(ax_mean)
        ax_rms = fig.add_subplot(fig_grid[1,iog-1])
        anode_rms_plots.append(ax_rms)
        # anode plots
        
        if iog in [1,2]:
            anode_xy(anode_rms_plots[iog-1], geo_mod0, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1], geo_mod0, d, 'mean', 150,iog)
        elif iog in [3,4]:
            anode_xy(anode_rms_plots[iog-1], geo_mod1, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1], geo_mod1, d, 'mean', 150,iog)
        elif iog in [5,6]:
            anode_xy(anode_rms_plots[iog-1], geo_mod2, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1], geo_mod2, d, 'mean', 150,iog)
        else:
            anode_xy(anode_rms_plots[iog-1], geo_mod3, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1], geo_mod3, d, 'mean', 150,iog)


    cbar_ax = fig.add_subplot(fig_grid[0,8])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=150),\
                                   cmap='viridis'), cax=cbar_ax,label='ADC Counts')

    cbar_ax2 = fig.add_subplot(fig_grid[1,8])
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=10),\
                                   cmap='viridis'), cax=cbar_ax2, label='ADC Counts')



    # summary plots
    #summary_ax = [sum_mean_ax,sum_rms_ax]
    #make_summary_plot(d,summary_ax)

    #plt.show()
    # outname = input_file.split('.json')[0]+'_2x2.png'
    print('Saving to: ', output_file)
    fig.savefig(output_file)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, \
                        type=str, help='''JSON-formatted dict of \
                        chip_key:channel''', required=True)
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
