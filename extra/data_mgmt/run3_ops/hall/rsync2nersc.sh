#!/usr/bin/env bash

src=$1; shift
dest=$1; shift

while true; do
    rsync -avhP "$src" "$dest"
    sleep 30
done
