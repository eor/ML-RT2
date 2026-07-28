#!/bin/sh
# Download the 053 training dataset (~2.5 GB) from Google Drive via gdown.
#
# This uses the same mechanism as the rest of the ML-RT project (see
# ML-RT/paper_data/pretrained_models.sh). The old astro.rug.nl mirror is gone.
#
# ONE-TIME SETUP (whoever hosts the data):
#   1. zip the five .npy files:  zip 053_data_set.zip data_*_profiles.npy data_parameters.npy
#   2. upload 053_data_set.zip to Google Drive; set sharing to "anyone with the link"
#   3. copy the file id from the share URL  (…/file/d/<FILEID>/view)
#   4. paste it into FILEID below (and commit)
#
# Requires: gdown (pip install gdown), unzip, md5sum (optional, for the integrity check).

BASEURL='https://drive.google.com/uc?id='
FILEID='REPLACE_WITH_GDRIVE_FILE_ID'      # <-- Google Drive id of 053_data_set.zip
ZIP='053_data_set.zip'

# expected md5 checksums of the extracted files (carried over from the original release)
CHECKSUMS="62375d3a72ee14f12e1d96a589d3c1de  data_HII_profiles.npy
bfbfbb279b41aed19c39d933633a67e3  data_T_profiles.npy
a8e1de335d896a7fcd573d401836a23e  data_parameters.npy
2c2bd60cbd837aaa42a9391d4c1d1a10  data_HeII_profiles.npy
0ed606f9bc78e175583372ae203c0cbc  data_HeIII_profiles.npy"

dl()      { gdown "$BASEURL$FILEID" -O "$ZIP"; }
extract() { unzip -o "$ZIP"; }
check()   {
    printf '%s\n' "$CHECKSUMS" | while read -r sum name; do
        [ -z "$name" ] && continue
        got=$(md5sum "$name" 2>/dev/null | awk '{print $1}')
        if [ "$sum" = "$got" ]; then echo "Checksum OK: $name"
        else echo "Checksum NOT ok: $name"; fi
    done
}

if [ "$FILEID" = "REPLACE_WITH_GDRIVE_FILE_ID" ]; then
    echo "Error: set FILEID to the Google Drive id of $ZIP first (see the comments). Exiting."
    exit 1
fi
if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown not found. Install it with 'pip install gdown'. Exiting."
    exit 1
fi

dl

if command -v unzip >/dev/null 2>&1; then extract
else echo "Warning: unzip not found; please extract $ZIP manually."; fi

if command -v md5sum >/dev/null 2>&1; then check
else echo "Warning: md5sum not found; skipping the integrity check."; fi
