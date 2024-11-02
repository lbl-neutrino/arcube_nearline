#!/usr/bin/env bash

default_data_root=/global/cfs/cdirs/dune/www/data/FSD
default_log_root=/pscratch/sd/d/dunepr/logs

data_root=${ARCUBE_NEARLINE_DATA_ROOT:-$default_data_root}
log_root=${ARCUBE_NEARLINE_LOG_ROOT:-$default_log_root}

data_outbase=$data_root/nearline/$stage
staging_outbase=$data_root/nearline/STAGING/$stage
plot_outbase=$data_outbase/plots
json_outbase=$data_outbase/jsons

log_outbase=$log_root/nearline/$stage
