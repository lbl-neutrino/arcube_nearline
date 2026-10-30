# CERN coldbox nearline

Lightweight nearline for the CERN coldbox CRP, run with
[workflow-runner](https://github.com/jaafar-chakrani/workflow-runner).
Nothing is uploaded anywhere: outputs are plain files on disk.

```
watch_latestfile.py                      # checks input_folder every --interval, runs the workflow on new files
plot_latestfile_workflow.yaml            # workflow: raw or packet file -> packet -> pedestal plot
packetize.sh                             # checks raw vs packet; raw -> packet file (larpix-control)
plot_panel_ped.sh                        # packet file -> pedestal PNG + JSON
plot_panel_ped/make_mean_std_json.py     # per-channel pedestal mean/std -> JSON
plot_panel_ped/make_crp_panel_ped.py     # JSON + geometry -> mean/RMS map of the 60 CRP tiles
geometry/multi_tile_layout_crp.yaml      # CRP geometry
```

## Environment

Python >= 3.10 (larpix-control uses `match`):

```bash
/opt/homebrew/bin/python3.12 -m venv ~/coldbox_env && source ~/coldbox_env/bin/activate
pip install numpy h5py matplotlib pyyaml
pip install "git+https://github.com/larpix/larpix-control.git@direct-convert-mixed-v2-v3"
pip install -e /path/to/workflow-runner
```

larpix-control must be the **`direct-convert-mixed-v2-v3`** branch: the CRP mixes
LArPix-v2 and v3 (io_groups 3,4 are v3) and older branches decode it wrongly.

## Running the nearline

Edit `input_folder` and `output_folder` in `plot_latestfile_workflow.yaml`, then:

```bash
./watch_latestfile.py plot_latestfile_workflow.yaml                 # check every minute
./watch_latestfile.py plot_latestfile_workflow.yaml --interval 30s  # or 5m, 1h, 90 (seconds)
```

Every interval it looks for `*.hdf5` / `*.h5` files in `input_folder` and processes each
new one once its size has not changed since the previous check and it has not been
modified for `stability_check_time` seconds. Files already in the folder when it
starts are skipped (unless they are still being written); to process those:

```bash
export COLDBOX_NEARLINE_DIR=/path/to/ndproto_nearline/cern_coldbox
workflow-runner batch plot_latestfile_workflow.yaml
workflow-runner process plot_latestfile_workflow.yaml /ABSOLUTE/path/to/X.hdf5
```

Outputs, for an input file `X.hdf5`:

```
<output_folder>/convert_to_packet/X.hdf5   # packet file (symlink to X.hdf5 if it already was one)
<output_folder>/panel_ped/X.png            # pedestal mean / RMS map
<output_folder>/panel_ped/X.json           # per-channel [mean, std]
```

## Scripts on their own

```bash
./packetize.sh INPUT_FILE PACKET_FILE
./plot_panel_ped.sh PACKET_FILE OUTPUT_PNG
```

- `packetize.sh` first checks what `INPUT_FILE` is, from its HDF5 content (not its name):
  - `packets` dataset: already a packet file, `PACKET_FILE` becomes a symlink to it;
  - `msgs` + `msg_headers`: raw file, converted;
  - anything else, or not readable as HDF5: the file fails with a message.

  Raw files get the same conversion as `justin/packetize_charge_data.jobscript`:
  `--asic-version 1:2,2:2,3:3,4:3,5:2,6:2 --format-version 3.0 --direct`
  (override the map with the `ASIC_VERSION` env variable).
- `plot_panel_ped.sh` uses data packets only (`packet_type == 1`, `valid_parity == 1`).
  Pixels with no data are grey; chips not in the geometry are skipped and listed.
  Color scales go up to the 99th percentile (set `--mean_max` / `--rms_max` in
  `make_crp_panel_ped.py`). Another geometry: `GEOMETRY_FILE` env variable.

## workflow-runner notes

- commands run inside `input_folder`: give `process` an **absolute** file path;
- paths are not quoted by workflow-runner: keep the quotes in the yaml commands;
- each step is killed after 5 minutes (a 1.7 GB raw file takes ~25 s for both steps).
