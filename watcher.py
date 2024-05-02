#!/usr/bin/env python

import argparse
from pathlib import Path
import time
from typing import Callable

from fireworks import Firework, LaunchPad, ScriptTask
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler


class EventHandler(FileSystemEventHandler):
    def __init__(self, watcher: "Watcher"):
        self.watcher = watcher

    def on_closed(self, event: FileSystemEvent):
        path = Path(event.src_path)
        self.watcher.maybe_make_firework(path)


class Watcher:
    def __init__(self, prog: Path, paths: list[Path], exts: list[str],
                 cond: Callable[[Path], bool]):
        self.prog = prog
        self.paths = paths
        self.exts = exts
        self.cond = cond

        self.seen_files = set()
        self.lpad = LaunchPad.auto_load()
        self.db = self.lpad.connection[self.lpad.name] # type: ignore

    def maybe_make_firework(self, p: Path):
        if ( (p in self.seen_files) or
             (not self.cond(p)) or
             self.find_firework(p) or
             (time.time() - p.stat().st_mtime < 300) ):
            # print(f'NO: {p}')
            return None

        script = f'{self.prog} {p}'
        task = ScriptTask.from_str(script)
        fw = Firework(task, name=self.prog.name)
        # NOTE: add_wf updates fw.fw_id to match the DB's value
        self.lpad.add_wf(fw)
        self.seen_files.add(p)
        print(f'FW {fw.fw_id:05}: {p}')
        return fw

    def find_firework(self, p: Path):
        q = {'spec._tasks.script': f'{self.prog} {p}'}
        return self.db['fireworks'].find_one(q)

    def snarf(self):
        for path in self.paths:
            for ext in self.exts:
                for p in path.rglob(f'*.{ext}'):
                    self.maybe_make_firework(p)

    def watch_inotify(self):
        handler = EventHandler(self)
        observer = Observer()
        observer.schedule(handler, self.path, recursive=True)
        observer.start()
        try:
            while observer.is_alive():
                observer.join(1)
        finally:
            observer.stop()
            observer.join()

    def watch_dumb(self):
        while True:
            self.snarf()
            time.sleep(300)



def is_hdf5(p: Path):
    return p.suffix.lower() in ['.h5', '.hdf5']


def is_raw_binary(p: Path):
    return is_hdf5(p) and p.name.startswith('binary')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prog', type=Path)
    ap.add_argument('--path', type=Path)
    ap.add_argument('--path-file')
    ap.add_argument('--require-binary', action='store_true')
    ap.add_argument('--ext', help='File extension (if not h5 / hdf5)')
    args = ap.parse_args()

    if args.path:
        assert not args.path_file
        paths = [args.path]
    if args.path_file:
        assert not args.path
        paths = [Path(l.strip())
                 for l in open(args.path_file).readlines()]

    exts = ['.h5', '.hdf5']
    cond = is_hdf5
    if args.require_binary:
        cond = is_raw_binary
    elif args.ext:
        exts = [args.ext]
        cond = lambda p: p.suffix.lower() == args.ext

    w = Watcher(args.prog.absolute(), paths, exts, cond)
    # w.snarf()
    # w.watch()
    w.watch_dumb()


if __name__ == '__main__':
    main()
