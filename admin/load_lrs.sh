# source me

if [[ "$LMOD_SYSTEM_NAME" == "perlmutter" ]]; then
    module load python/3.13
fi

source _install/nearline_lrs.venv/bin/activate

export FW_CONFIG_FILE=$(realpath config/FW_config.yaml)

export ROOT_OF_ARCUBE_NEARLINE_LRS=$PWD
export PYTHONPATH=$ROOT_OF_ARCUBE_NEARLINE_LRS:$PYTHONPATH

export HDF5_USE_FILE_LOCKING=FALSE
