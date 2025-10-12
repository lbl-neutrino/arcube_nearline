#!/usr/bin/env python3

from hashlib import md5
from pathlib import Path
import os
import time

from PIL import Image

PATH_OF_PNG = '/home/admin/FSDVibMon/new_main/fft.png'
USERNAME = 'admin'
IP_OF_PI = '192.168.197.97'
DESTINATION = '/data/vibMon_plots'
SLEEP_SECONDS = 60


def fetch():
    src = f'{USERNAME}@{IP_OF_PI}:{PATH_OF_PNG}'
    dest = f'{DESTINATION}/tmp/fft.png'
    os.system(f'scp -i ~/.ssh/vibe_mon {src} {dest}')

    # do we have a valid png?
    try:
        with Image.open(dest) as im:
            pass
    except:
        return

    latest = f'{DESTINATION}/plots/fft.latest.png'

    if Path(latest).exists():
        with open(latest, 'rb') as f:
            last_hash = md5(f.read()).hexdigest()
        with open(dest, 'rb') as f:
            new_hash = md5(f.read()).hexdigest()
        if new_hash == last_hash:
            return

    today = time.strftime('%Y_%m_%d')
    now = time.strftime('%Y_%m_%d_%H_%M_%S_%Z')

    os.system(f'mkdir -p {DESTINATION}/plots/{today}')
    os.system(f'mv {dest} {DESTINATION}/plots/{today}/fft.{now}.png')
    os.system(f'cp {DESTINATION}/plots/{today}/fft.{now}.png {DESTINATION}/plots/fft.latest.png')

    if Path('mirror.sh').exists():
        os.system(f'./mirror.sh')


def main():
    os.system(f'mkdir -p {DESTINATION}/plots')
    os.system(f'mkdir -p {DESTINATION}/tmp')

    while True:
        fetch()
        time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    main()
