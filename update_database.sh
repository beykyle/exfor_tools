#!/bin/bash

path_to_zipfile=$1

# Check for help flag
if [[ "$*" == *"-h"* || "$*" == *"--help"* ]]; then
  echo "Usage: script.sh [path_to_zipfile] [--db-dir directory]"
  echo "Options:"
  echo "  -h, --help        Show this help message"
  echo "  --db-dir          Specify the directory where the database should be placed"
  exit 0
fi

year=$(echo "$path_to_zipfile" | grep -oP 'exfor-\K\d{4}')

# where to put data base
db_dir=$(echo "$@" | grep -oP '(?<=--db-dir\s)[^\s]+')
if [ -z "$db_dir" ]; then
  db_dir=$(dirname "$path_to_zipfile")
else
  mv "$path_to_zipfile" "$db_dir"
  path_to_zipfile="$db_dir/$(basename "$path_to_zipfile")"
fi

python3 x4i3_tools/setup_exfor_db.py  --exfor-master "$path_to_zipfile"
python3 x4i3_tools/setup_exfor_db.py --exfor-master "$path_to_zipfile" --rename="X4-$year-12-31" --create-x4i3-tarfile

export X43I_DATAPATH="$db_dir/unpack_exfor-$year/X4-$year-12-31"
echo "setting 'X43I_DATAPATH' to $X43I_DATAPATH"
