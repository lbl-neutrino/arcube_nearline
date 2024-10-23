# arcube_nearline

Nearline prompt processing workflows for DUNE's ArgonCube 2x2 demonstrator.

## Installation

``` bash
admin/install.sh
```

The installation will go into `_install`. After installing, edit `config/my_launchpad.yaml` to configure the MongoDB connection. If it's a new database, you will need to run `lpad reset`.

## Environment setup

``` bash
source admin/load.sh
```

If your `my_launchpad.yaml` points to a new Mongo database, run `lpad reset` to initialize it.

## Monitoring the filesystem and loading the DB

``` bash
./watcher.py actions/packetize.sh --path /global/cfs/cdirs/dune/www/data/2x2/CRS/commission/April2024
```

This will continue to run in the foreground.

## Launching a worker loop

``` bash
cd $(mktemp -d)
rlaunch rapidfire
```

The temporary directory avoids littering the filesystem. Use `rlaunch rapidfire --nlaunches infinite` to prevent the worker from exiting when there's no available work.
