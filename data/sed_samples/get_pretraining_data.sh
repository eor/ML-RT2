#!/bin/sh

# GitHub has a 100MB filesize limit at this time, so please download the data using this script.

BASEURL='https://drive.google.com/uc?id='

FILE1='1IMf50ttdlsVS6OnpO3YVv32W55dBMsWn'
FILE1NAME='pretraining.npy.npz'
CHECKSUM1='7fe1048f4a4ed3b0ed2f19c89fafeabc'

dl() {

        gdown $BASEURL$FILE1
}
  

check() {

        VAR1=$(md5sum ${FILE1NAME} | awk '{print $1}')

        # 1
        if [ "$CHECKSUM1" = "$VAR1" ]; then

            echo "Checksum OK: "$FILE1NAME
            
        else
        
            echo "Checksum NOT ok: "$FILE1NAME       
            
        fi
        
}

if ! (command -v gdown &> /dev/null)
then
    echo "Error: gdown could not be found. Exiting"
    exit

else
    dl
fi


if ! (command -v md5sum &> /dev/null)
then
    echo "Warning: md5sum could not be found. Can't verify file integrity"
    exit
    
else
    check
fi

