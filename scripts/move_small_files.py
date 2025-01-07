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
    ap.add_argument('--min-size', type=float, default=0.1,
                    help='Min file size (in GiB) to keep in INDIR')
    args = ap.parse_args()

    for jsonpath in args.indir.rglob('*.json'):
        datapath = jsonpath.parent / jsonpath.stem

        if not datapath.exists():
            print(f'INVALID {jsonpath}\n')
            continue

        size_G = datapath.stat().st_size / 2**30

        if size_G >= args.min_size:
            continue

        print(f'MOVE {datapath}')

        out_datapath = args.outdir / datapath.relative_to(args.indir)
        out_jsonpath = out_datapath.with_suffix(out_datapath.suffix + '.json')

        os.makedirs(out_datapath.parent, exist_ok=True)

        shutil.move(datapath, out_datapath)
        shutil.move(jsonpath, out_jsonpath)


if __name__ == '__main__':
    main()
