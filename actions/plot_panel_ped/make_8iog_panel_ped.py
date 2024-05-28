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


centerline_chip_id=list(range(20,120,10))
horizontal_chip_id=list(range(11,21,1))+list(range(101,111,1))

centerline_channel_id=[60,52,53,48,45,41,35]
horizontal_channel_id=[60,61,62,63,1,2,3]+[35,34,33,31,30,29,28]

def tile_to_io_channel(tile):
    io_channel=[]
    for t in tile:
        for i in range(1,5,1): io_channel.append( ((t-1)*4)+i )
    return io_channel

def load_yaml(geometry_path):
    with open(geometry_path) as fi:
        geo = yaml.full_load(fi)
        chip_pix = dict([(chip_id, pix) for chip_id,pix in geo['chips']])
        vlines=np.linspace(-1*(geo['width']/2), geo['width']/2, 11)
        hlines=np.linspace(-1*(geo['height']/2), geo['height']/2, 11)    
    return geo, chip_pix, vlines, hlines

def anode_xy(ax, geo, chip_pix, vertical_lines, horizontal_lines, d, \
             metric, normalization, iog):
    if metric=='mean': metric=0
    if metric=='std': metric=1
     
    tile_dy = abs(max(vertical_lines))+abs(min(vertical_lines))
    tile_y_placement = [tile_dy*i for i in range(5)]
    mid_y = tile_y_placement[2]
    tile_y_placement = [typ-mid_y for typ in tile_y_placement]
    mid_vl = tile_dy/2.
    chip_y_placement = [typ+vl+mid_vl for typ in tile_y_placement[0:-1] \
                        for vl in vertical_lines[0:-1]]
    
    tile_dx = abs(max(horizontal_lines))+abs(min(horizontal_lines))
    tile_x_placement = [tile_dx*i for i in range(3)]
    mid_x = tile_x_placement[1]
    tile_x_placement = [txp-mid_x for txp in tile_x_placement]
    mid_hl = tile_dx/2.
    chip_x_placement = [txp+hl+mid_hl for txp in tile_x_placement[0:-1] \
                        for hl in horizontal_lines[0:-1]]
    
    #print('iog =',iog)
    #print('tile_dx =',tile_dx)
    #print('tile_dy =',tile_dy)

    if metric == 0: ax.set_title('io_group '+str(iog)+" Mean")
    elif metric == 1: ax.set_title('io_group '+str(iog)+" RMS")
#    ax.set_xlim(tile_x_placement[0]*1.01,tile_x_placement[-1]*1.01)
#    ax.set_ylim(tile_y_placement[0]*1.01,tile_y_placement[-1]*1.01)
#
#    for typ in tile_y_placement:
#        ax.hlines(y=typ, xmin=tile_x_placement[0],
#                     xmax=tile_x_placement[-1], colors=['k'], \
#                     linestyle='solid')
#
#    for txp in tile_x_placement:
#        ax.vlines(x=txp, ymin=tile_y_placement[0],
#                     ymax=tile_y_placement[-1], colors=['k'], \
#                     linestyle='solid')
#
#    for cyp in chip_y_placement:
#        ax.hlines(y=cyp, xmin=tile_x_placement[0], \
#                     xmax=tile_x_placement[-1], colors=['k'], \
#                     linestyle='dotted')
#    for cxp in chip_x_placement:
#        ax.vlines(x=cxp, ymin=tile_y_placement[0], \
#                     ymax=tile_y_placement[-1], colors=['k'], \
#                     linestyle='dotted') 

    pitch=4.4 # mm
    grid_tpc = np.zeros((7*10*4,7*10*2))
    if iog in [5,6]:
        grid_tpc = np.zeros((8*10*4,8*10*2))
        pitch = 3.87975

    displacement={1:(-0.5,1.5), 2:(0.5,1.5), 3:(-0.5,0.5), 4:(0.5, 0.5), \
                  5:(-0.5,-0.5), 6:(0.5,-0.5), 7:(-0.5,-1.5), 8:(0.5,-1.5)}
    shift_g = {1:(0,3),2:(1,3),3:(0,2),4:(1,2),5:(0,1),6:(1,1),7:(0,0),8:(1,0)}
    for i in range(1,9,1):
        io_channels=tile_to_io_channel([i])
        for chipid in chip_pix.keys():
            x,y = [[] for i in range(2)]
            for j in [iog]:
                for ioc in io_channels:
                    chip_key=str(j)+'-'+str(ioc)+'-'+str(chipid)
                    if chip_key not in d: continue
                    for channelid in range(64):
                        if channelid in [6,7,8,9,22,23,24,25,38,39,40,54,55,56,57] and iog not in [5,6]: continue
                        if d[chip_key][channelid][0]==-1: continue
                        weight=d[chip_key][channelid][metric]/normalization
                        if i%2!=0:
                            xc=geo['pixels'][chip_pix[chipid][channelid]][1]
                            yc=geo['pixels'][chip_pix[chipid][channelid]][2]*-1
                        if i%2==0:
                            xc=geo['pixels'][chip_pix[chipid][channelid]][1]*-1
                            yc=geo['pixels'][chip_pix[chipid][channelid]][2]
                        x_g=int((xc+tile_dx/2-pitch/2)/pitch)+70*shift_g[i][0]
                        y_g=int((yc+tile_dy/2-pitch/2)/pitch)+70*shift_g[i][1]
                        if iog in [5,6]:
                            x_g=int((xc+tile_dx/2)/pitch)+80*shift_g[i][0]
                            y_g=int((yc+tile_dy/2)/pitch)+80*shift_g[i][1]
                        if weight>1.: weight=1.0
                        grid_tpc[y_g][x_g] = weight
                        cmap = cm.get_cmap('viridis')

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
    geo, chip_pix, vertical_lines, horizontal_lines = load_yaml('layout-2.4.0.yaml')
    geo_v2b, chip_pix_v2b, vertical_lines_v2b, horizontal_lines_v2b = load_yaml('layout-2.5.1.yaml')

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
        if iog in [5,6]:
            anode_xy(anode_rms_plots[iog-1],geo_v2b, chip_pix_v2b, vertical_lines_v2b, horizontal_lines_v2b, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1],geo_v2b, chip_pix_v2b, vertical_lines_v2b, horizontal_lines_v2b, d, 'mean', 150,iog)
        else:
            anode_xy(anode_rms_plots[iog-1],geo, chip_pix, vertical_lines, horizontal_lines, d, 'std', 10,iog)
            anode_xy(anode_mean_plots[iog-1],geo, chip_pix, vertical_lines, horizontal_lines, d, 'mean', 150,iog)


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
