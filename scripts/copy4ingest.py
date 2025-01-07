#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()

    for jsonpath in args.indir.rglob('*.json'):
        datapath = jsonpath.parent / jsonpath.stem

        with open(jsonpath) as f:
            meta = json.load(f)

        assert meta['name'] == datapath.name

        # .h5 -> .hdf5, if applicable
        suffix = datapath.suffix
        if suffix == '.h5':
            suffix = '.hdf5'
        meta['name'] = datapath.with_suffix(suffix).name

        out_datapath = \
            args.outdir / datapath.relative_to(args.indir).with_suffix(suffix)
        out_jsonpath = out_datapath.with_suffix(out_datapath.suffix + '.json')

        if out_datapath.exists() and out_jsonpath.exists():
            print(f'SKIP {out_datapath}\n')
            continue

        os.makedirs(out_datapath.parent, exist_ok=True)
        print(f'COPY {datapath} --> {out_datapath}')
        shutil.copy(datapath, out_datapath)

        print(f'WRITE {out_jsonpath}\n')
        with open(out_jsonpath, 'w') as f:
            json.dump(meta, f, indent=4)


if __name__ == '__main__':
    main()
