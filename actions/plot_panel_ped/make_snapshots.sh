#!/bin/bash

INPUT_FILES='selftrig_04Apr2024/summary/packet-*.json'

for FILE in ${INPUT_FILES}
do
  echo ${FILE}
  python make_8iog_panel_ped.py --input_file ${FILE}
done
