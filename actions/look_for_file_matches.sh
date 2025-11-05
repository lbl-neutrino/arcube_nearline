#!/bin/bash

runsdb=/global/cfs/cdirs/dune/www/data/2x2/DB/RunsDB/latest/run2/2x2runs_run2.latest.sqlite
#charge_dir=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_charge_low_energy/
#light_dir=/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light_low_energy/
charge_dir=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/run2/crs_flowed
light_dir=/global/cfs/cdirs/dune/users/sfogarty/flow_LowEnergy/2x2/run2/lrs_flowed/
cd ..
python3 look_for_file_matches.py --runsdb_path $runsdb --charge_dir_path $charge_dir --light_dir_path $light_dir --text_file_dirname 'lowE_matching_inputs'