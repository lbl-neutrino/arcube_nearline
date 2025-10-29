#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import zlib


def get_checksum(path: Path):
    cksum = 1
    with open(path, 'rb') as f:
        chunksize = int(1e9)
        while data := f.read(chunksize):
            cksum = zlib.adler32(data, cksum)
    return cksum & 0xffffffff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcdir', type=Path, required=True)
    ap.add_argument('--destdir', type=Path, required=True)
    args = ap.parse_args()

    for p in args.destdir.iterdir():
        if not p.is_file():
            continue

        p_json = args.srcdir / (p.name + '.json')
        if not p_json.is_file():
            continue

        with open(p_json) as f:
            metadata = json.load(f)

        cksum = get_checksum(p)
        metadata['checksums']['adler32'] = f'{cksum:08x}'
        metadata['size'] = p.stat().st_size

        p_json_out = args.destdir / p_json.name
        with open(p_json_out, 'w') as f_out:
            json.dump(metadata, f_out, indent=4)

        p_json.rename(p_json.with_suffix(p_json.suffix + '.moved'))

        print(p)


if __name__ == '__main__':
    main()
