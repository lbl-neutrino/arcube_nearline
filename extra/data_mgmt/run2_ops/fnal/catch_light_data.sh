#!/usr/bin/env bash

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
NC=$(tput sgr0)

while true; do
  for f in $(find . -name '*.data' -size +1k -mmin +2); do
    if [[ ! -f "$f.json" ]]; then
      continue
    fi
    echo "${GREEN}Moving ${BLUE}$f${NC}"
    ~/mkramer/run2_ops/move_light_data.sh $f
    sleep 1
  done
  date
  sleep 60
done
