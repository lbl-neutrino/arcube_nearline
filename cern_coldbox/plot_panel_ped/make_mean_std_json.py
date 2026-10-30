#!/usr/bin/env python3
# Per-channel pedestal mean and std from a LArPix packet file.
# Output JSON: {"iog-ioc-chip": [[mean, std] x 64 channels]}, [-1, -1] = no data.
import argparse
import json

import h5py
import numpy as np

FIELDS = ['io_group', 'io_channel', 'chip_id', 'channel_id', 'packet_type', 'valid_parity', 'dataword']
BLOCK = 10_000_000  # packets read at a time


def main(input_file, output_file):
    keys, n, s, s2 = [], [], [], []
    with h5py.File(input_file, 'r') as f:
        dset = f['packets']
        for start in range(0, len(dset), BLOCK):
            p = dset.fields(FIELDS)[start:start + BLOCK]
            # coldbox data packets are packet_type 1 (the 2x2 nearline used 0)
            p = p[(p['valid_parity'] == 1) & (p['packet_type'] == 1)]
            # one integer per (io_group, io_channel, chip_id, channel_id)
            key = (((p['io_group'].astype(np.int64) << 8 | p['io_channel']) << 8
                    | p['chip_id']) << 8 | p['channel_id'])
            k, inv = np.unique(key, return_inverse=True)
            adc = p['dataword'].astype(float)
            keys.append(k)
            n.append(np.bincount(inv))
            s.append(np.bincount(inv, weights=adc))
            s2.append(np.bincount(inv, weights=adc**2))

    # merge the blocks
    k, inv = np.unique(np.concatenate(keys), return_inverse=True)
    n = np.bincount(inv, weights=np.concatenate(n))
    mean = np.bincount(inv, weights=np.concatenate(s)) / n
    std = np.sqrt(np.maximum(np.bincount(inv, weights=np.concatenate(s2)) / n - mean**2, 0))

    output = {}
    for key, m, sd in zip(k, mean, std):
        iog, ioc, cid, chan = key >> 24, key >> 16 & 255, key >> 8 & 255, key & 255
        chip_key = '{}-{}-{}'.format(iog, ioc, cid)
        if chip_key not in output:
            output[chip_key] = [(-1, -1)] * 64
        output[chip_key][chan] = (m, sd)

    print('Channels with data: {}'.format(len(k)))
    with open(output_file, 'w') as fout:
        json.dump(output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='''HDF5 packet file''')
    parser.add_argument('--output_file', required=True, type=str, help='''JSON output file''')
    args = parser.parse_args()
    main(**vars(args))
