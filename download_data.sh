#!/bin/bash
set -e

FILE_ID="1cQqwalPocBCxhmEWhai10YQJv81tzPpc"
OUTPUT_TAR="embedded_data.tar.gz"

echo "===== Embedded data Download Starting... ====="

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O ${OUTPUT_TAR}
mkdir embedded_data
tar -xzvf ${OUTPUT_TAR} -C embedded_data
rm ${OUTPUT_TAR}

echo "===== Embedded data Download Complete! ====="
