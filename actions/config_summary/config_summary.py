#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import argparse
from datetime import datetime
import numpy as np
import os
import json
import re
from collections import defaultdict
import tqdm
import io
from pathlib import Path
import tarfile
import shutil
import math

plt.rcParams.update({'font.size': 15}) # Sets the global default font size

def date_from_ped_filename(filename):
    date_str = filename.split('-')[-1].split('_CDT')[0]
    timestamp = datetime.strptime(date_str,"%Y_%m_%d_%H_%M_%S")
    return timestamp

def main(input_file, output_file):

    # Fields to skip in ASIC configs
    skip_fields = {
        "meta",
        "enable_miso_upstream",
        "enable_miso_downstream",
        "enable_miso_differential",
        "enable_mosi",
        "enable_piso_upstream",
        "enable_piso_downstream",
        "enable_posi",
        "ref_current_trim"
        
    }
    
    # Regex patterns to skip in ASIC configs
    skip_patterns = [
        r"tx_slices\d+",
        r"r_term\d+",
        r"i_tx_diff\d+",
    ]
    
    # Compile regex patterns
    compiled_skip_patterns = [re.compile(p) for p in skip_patterns]

    print('Opening file: ', input_file)
    h5f = h5py.File(input_file, 'r')
    print('File opened.')

    # Untar ASIC configs in current directory
    print('Extracting ASIC configs...')
    buf = io.BytesIO(np.array(h5f['daq_configs']).data)

    with tarfile.open(fileobj=buf) as tarf:
        name = tarf.getmembers()[0].name
        if Path(name).exists():
            msg = f'Directory {name} already exists; not extracting, sorry'
            # raise RuntimeError(msg)
        
        # select only asic_configs for slightly faster untar
        subdir_and_files = [ tarinfo for tarinfo in tarf.getmembers() if "asic_configs" in tarinfo.name ]
        # print(subdir_and_files)
        tarf.extractall(members=subdir_and_files)

    print('Extracted.')
    # Group Values by Field and by io_group
    pattern = re.compile(r"config_(\d+)-\d+-\d+\.json")
    
    # field_data[field][io_group] = list of values
    field_data = defaultdict(lambda: defaultdict(list))
    
    count_v2a_chips = 0
    count_v2b_chips = 0
    
    count_disabled_channels = 0

    asic_config_folder = os.path.join(name, 'asic_configs')

    print('Looping over the ASIC configs...')
    for filename in tqdm.tqdm(os.listdir(asic_config_folder)):
        match = pattern.match(filename)
        if match:
            io_group = int(match.group(1))
            filepath = os.path.join(asic_config_folder, filename)
            with open(os.path.join(asic_config_folder, filename), 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    # Skip by exact field name
                    if key in skip_fields:
                        continue
                    # Skip by pattern
                    if any(pat.match(key) for pat in compiled_skip_patterns):
                        continue
                    if isinstance(value, list):
                        field_data[key][io_group].extend(value)
                    else:
                        field_data[key][io_group].append(value)
                
                if io_group == 5 or io_group == 6:
                    count_v2b_chips += 1
                    if 'channel_mask' in data.keys():
                        count_disabled_channels += data['channel_mask'].count(1)
                else:
                    count_v2a_chips += 1
                    if 'channel_mask' in data.keys():
                        count_disabled_channels += data['channel_mask'].count(1) - 15

    # Remove ASIC configs
    shutil.rmtree(name, ignore_errors=True)
    
    # Count number of disabled channels
    count_disabled_channels += (6 * 8 * 100 - count_v2a_chips) * 49 + (2 * 8 * 100 - count_v2b_chips) * 64

    # Plot everything
    print('Plotting...')
    num_fields = len(field_data)
    cols = 4
    rows = math.ceil(num_fields / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    colors = plt.get_cmap("tab10")  # For coloring different io_group values
    
    for i, (field, a_groups) in enumerate(field_data.items()):
        ax = axes[i]
        
        # Plot each group for this field
        for j, (io_group, values) in enumerate(sorted(a_groups.items())):
            avg_val = sum(values) / len(values) if values else 0
            label = f"io_group={io_group} (avg={avg_val:.2f})"
            ax.hist(values, bins=16, histtype='step', label=label, color=colors(j % 8))
    
        ax.set_xlabel(field)
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_yscale('log')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'{os.path.basename(input_file)}\nNumber of configured chips: {count_v2a_chips+count_v2b_chips}\nNumber of disabled channels: {count_disabled_channels}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    # plt.close()
    
    print(f"Saved figure to: {output_file}")
  
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, \
                        type=str, help='''Path to packet-formatted hdf5 file''', required=True)
    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
