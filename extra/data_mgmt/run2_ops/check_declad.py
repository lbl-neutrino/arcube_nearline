#!/usr/bin/env python3

import argparse
import subprocess
from subprocess import call, DEVNULL
import sys

SCOPE = None

def call_and_capture(args, **kwargs):
    """
    A drop-in replacement for subprocess.call that also captures stdout/stderr
    without preventing them from being printed to the terminal.

    Parameters:
        args (list or str): Command and arguments.
        kwargs: Extra keyword arguments passed to subprocess.Popen.

    Returns:
        exit_code (int), stdout_str (str), stderr_str (str)
    """

    # Force text mode so we get strings instead of bytes
    kwargs.setdefault("text", True)

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs
    )

    captured_stdout = []
    captured_stderr = []

    # Read stdout and stderr in parallel
    # This simple loop works well for commands that output intermittently
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()

        if stdout_line:
            sys.stdout.write(stdout_line)
            sys.stdout.flush()
            captured_stdout.append(stdout_line)

        if stderr_line:
            sys.stderr.write(stderr_line)
            sys.stderr.flush()
            captured_stderr.append(stderr_line)

        if not stdout_line and not stderr_line and process.poll() is not None:
            break

    exit_code = process.wait()
    return exit_code, "".join(captured_stdout), "".join(captured_stderr)


def check_metacat(fname):
    cmd = f'metacat file show {fname}'
    return 0 == call(cmd, shell=True) # stdout=DEVNULL, ...


def check_rucio(fname):
    cmd = f'rucio replica list file {fname}'
    ret, stdout, stderr = call_and_capture(cmd, shell=True)
    return 'not found' not in stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file_list_file')
    ap.add_argument('-s', '--scope', default='neardet-2x2-lar-light')
    args = ap.parse_args()

    with open(args.file_list_file) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        line = line.replace('.copied', '')
        if line.endswith('.json'):
            continue

        fname = args.scope + ':' + line.strip()
        metacat_good = check_metacat(fname)
        rucio_good = check_rucio(fname)

        if not metacat_good:
            print(f'BAD METACAT: {line}')

        if not rucio_good:
            print(f'BAD RUCIO: {line}')

        if metacat_good and rucio_good:
            print(f'ALL GOOD: {line}')


if __name__ == '__main__':
    main()
