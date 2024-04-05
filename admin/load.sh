# source me

if [[ "$LMOD_SYSTEM_NAME" == "perlmutter" ]]; then
    module load python/3.11
fi

source _install/nearline.venv/bin/activate

export FW_CONFIG_FILE=$(realpath config/FW_config.yaml)

export ROOT_OF_ARCUBE_NEARLINE=$PWD
