#!/usr/bin/env bash

default_data_root=/global/cfs/cdirs/dune/www/data/2x2
default_log_root=/pscratch/sd/d/dunepr/logs

data_root=${ARCUBE_NEARLINE_DATA_ROOT:-$default_data_root}
log_root=${ARCUBE_NEARLINE_LOG_ROOT:-$default_log_root}

nearline_name=nearline_run2
nearline_root=$data_root/$nearline_name
data_outbase=$nearline_root/$stage
plot_outbase=$data_outbase/plots
json_outbase=$data_outbase/jsons

log_outbase=$log_root/$nearline_name/$stage
