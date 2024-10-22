#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from datetime import datetime
import matplotlib.dates as mdates
import yaml
from copy import copy
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
import argparse
from collections import defaultdict
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _default_pxy():
    return (0., 0.)

def _rotate_pixel(pixel_pos, tile_orientation):
    return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]


def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def unique_to_channel_id(unique):
    return unique % 100

def unique_to_chip_id(unique):
    return (unique // 100) % 1000

def unique_to_io_channel(unique):
    return (unique//(100*1000)) % 1000

def unique_to_tiles(unique):
    return ((unique_to_io_channel(unique)-1) // 4) + 1

def unique_to_io_group(unique):
    return (unique // (100*1000*1000)) % 1000


def parse_file(filename, max_entries):
    d = dict()
    f = h5py.File(filename, 'r')
    unixtime = f['packets'][:]['timestamp'][f['packets']
                                            [:]['packet_type'] == 4]
    livetime = np.max(unixtime)-np.min(unixtime)
    data_mask = f['packets'][:]['packet_type'] == 0
    valid_parity_mask = f['packets'][:]['valid_parity'] == 1
    mask = np.logical_and(data_mask, valid_parity_mask)
    adc = f['packets']['dataword'][mask][:max_entries]
    unique_id = unique_channel_id(f['packets'][mask][:max_entries])
    unique_id_set = np.unique(unique_id)
    chips = f['packets']['chip_id'][mask][:max_entries]

    print("Number of packets in parsed files =", len(unique_id))
    for chip in range(11, 171):
        _iomask = chips == chip
        _adc = adc[_iomask]
        _unique_id = unique_id[_iomask]
        for i in set(_unique_id):
            id_mask = _unique_id == i
            masked_adc = _adc[id_mask]
            d[i] = dict(
                mean=np.mean(masked_adc),
                std=np.std(masked_adc),
                rate=len(masked_adc) / (livetime + 1e-9))
    return d


def load_multitile_geometry(geometry_path):
    
    geo = None
    
    with open(geometry_path) as fi:
        geo = yaml.full_load(fi)

    # Adapted from: https://github.com/larpix/larpix-v2-testing-scripts/blob/master/event-display/evd_lib.py

    pixel_pitch = geo['pixel_pitch']

    chip_channel_to_position = geo['chip_channel_to_position']
    tile_orientations = geo['tile_orientations']
    tile_positions = geo['tile_positions']
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
            x += tile_positions[tile][2] 
            y += tile_positions[tile][1] 

            geometry[(io_group, io_group_io_channel_to_tile[(
                io_group, io_channel)], chip, channel)] = x, y

    return geometry, pixel_pitch


def anode_xy(ax, geo, pitch, d, metric, normalization):
    
    cmap = matplotlib.colormaps['viridis']
        
    xmin = min(np.array(list(geo.values()))[:, 0])-pitch/2
    xmax = max(np.array(list(geo.values()))[:, 0])+pitch/2
    ymin = min(np.array(list(geo.values()))[:, 1])-pitch/2
    ymax = max(np.array(list(geo.values()))[:, 1])+pitch/2
    
    uniques = np.array(list(d.keys()))
    
    for tpc in range(1, 3):

        io_groups = [1, 2] if tpc == 1 else [3, 4]

#         ax[tpc-1].set_xlabel('X Position [mm]')
#         ax[tpc-1].set_ylabel('Y Position [mm]')

        ax[tpc-1].set_xlim(xmin*1.05, xmax*1.05)
        ax[tpc-1].set_ylim(ymin*1.05, ymax*1.05)

        ax[tpc-1].set_aspect('equal')

        for io_group in io_groups:

            mask = unique_to_io_group(uniques) == io_group

            print('Getting {} for io_group {}'.format(metric, io_group))
            d_keys = uniques[mask]
            print('\tNumber of channels: ', len(d_keys))

            a, b = geo[(io_group, 10*(io_group-1)+1, 71, 1)]

            ax[tpc-1].text(a, b+25,
                           f'io_group = {io_group}', ha='center')

            for key in d_keys:
                channel_id = unique_to_channel_id(key)
                chip_id = unique_to_chip_id(key)
                tile = unique_to_tiles(key) + 10 * (io_group - 1)

                if chip_id not in range(11, 171):
                    continue
                if channel_id not in range(64):
                    continue

                x, y = geo[(io_group, tile, chip_id, channel_id)]

                weight = d[key][metric]/normalization

                if weight > 1.0:
                    weight = 1.0

                r = Rectangle((x-(pitch/2.), y-(pitch/2.)),
                              pitch, pitch, color=cmap(weight))
                ax[tpc-1].add_patch(r)


def date_from_filename(filename):
    #date_str = filename.split('-')[-1].split('.json')[0]
    #date_str = filename.split('-')[-1].split('CET.json')[0]
    date_str = filename.split('-')[-1].split('_CEST.h5')[0]
    #timestamp = datetime.fromisoformat(date_str)
    print(date_str)
    #timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M%Z")
    timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M_%S")
    return timestamp

def main(input_file, output_file):
    
    d = parse_file(input_file, max_entries=-1)

    timestamp = date_from_filename(input_file)

    geo, pitch = load_multitile_geometry('layouts_fsd/multi_tile_layout-3.0.40_fsd.yaml')

    fig, ax = plt.subplots(2, 2, figsize=(15, 25))
    fig.suptitle(timestamp)

    max_mean = 150
    anode_xy(ax[0], geo, pitch, d, 'mean', max_mean)
    ax[0, 0].set_title('TPC 1 - Mean')
    ax[0, 1].set_title('TPC 2 - Mean')
    
    divider = make_axes_locatable(ax[0, 1])
    cax0 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_mean),
                                   cmap='viridis'), cax=cax0,label='ADC Counts')


    max_std = 10    
    anode_xy(ax[1], geo, pitch, d, 'std', max_std)
    ax[1, 0].set_title('TPC 1 - Std. Dev.')
    ax[1, 1].set_title('TPC 2 - Ste. Dev.')

    divider = make_axes_locatable(ax[1, 1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_std),
                                   cmap='viridis'), cax=cax1,label='ADC Counts')

    

    plt.tight_layout()
    
    print('Saving to: ', output_file)
    fig.savefig(output_file)
    print('Saved.')

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
