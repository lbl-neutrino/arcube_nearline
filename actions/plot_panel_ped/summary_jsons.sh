#!/bin/bash
#INPUT_FILES='/global/cfs/cdirs/dune/www/data/2x2/CRS/commission/April2024/continued_pedestal/packet/packet-*'
#INPUT_FILES='/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/April2024/continued_pedestal/packet-*'
INPUT_FILES='/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/April2024/continued_selftrigger/shifter_test1/packet-*'

for FILE in ${INPUT_FILES}
do
  #echo ${FILE}
  OUTNAME="${FILE##*/}"
  OUTNAME=`echo ${OUTNAME} | sed 's/_\(...\).h5/_\1/'`
  OUTNAME='selftrig_04Apr2024/summary/'${OUTNAME}''
  # don't do it again if it's already been done
  if [[ -f "${OUTNAME}".json ]]; then
    echo "${OUTNAME} exists. Skipping."
    continue
  fi
  echo 'working on' ${OUTNAME}
  python make-mean-std-json.py --input_file ${FILE} --file_prefix ${OUTNAME}
  #python make-mean-std-json.py --input_file ${FILE} --file_prefix 'summary'
done
