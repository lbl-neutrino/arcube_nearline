#!/usr/bin/env bash

set -o errexit

if [[ "$LMOD_SYSTEM_NAME" == "perlmutter" ]]; then
    module load python/3.11
fi

basedir=$PWD

mkdir _install
cd _install

python -m venv nearline.venv
source nearline.venv/bin/activate

pip install --upgrade pip wheel setuptools

# FireWorks currently uses the deprecated safe_load function
pip install "ruamel.yaml<0.18.0"
# FireWorks apparently needs pytest at runtime
pip install pytest

git clone https://github.com/lbl-neutrino/fireworks.git
pushd fireworks
# editable install doesn't work for fireworks atm
pip install ".[rtransfer,newt,daemon_mode,flask-plotting,workflow-checks,graph-plotting]"
popd

pip install watchdog

git clone https://github.com/larpix/larpix-control.git
pushd larpix-control
pip install --editable .
popd

# For Kevin's plots
pip install pyyaml

# pyyaml-include 2.x has breaking changes
pip install pyyaml-include==1.3.2

# install h5flow
git clone -b main https://github.com/larpix/h5flow.git
cd h5flow
pip install -e .
cd ..

# install ndlar_flow
git clone -b develop https://github.com/larpix/ndlar_flow.git
cd ndlar_flow
pip install -e .
cd scripts/proto_nd_scripts
./get_proto_nd_input.sh
cd ../../..

pip install adc64format

# For light DQM, etc.:
pip install awkward

cd "$basedir"

confdir=$(realpath config)
sed "s%{CONFIG_FILE_DIR}%$confdir%" config/templates/FW_config.template.yaml \
    > config/FW_config.yaml

if [[ ! -e config/my_launchpad.yaml ]]; then
    cp config/templates/my_launchpad.template.yaml config/my_launchpad.yaml
fi

printf "\nRemember to edit the settings in config/my_launchpad.yaml\n"
