#!/bin/sh

# GitHub has a 100MB filesize limit at this time, so please download the data using this script.

BASEURL='https://drive.google.com/uc?id='

FILE1='1D690uoF9VKzcjnWW0wDev4qceFjGawKN'
FILE1NAME='data_pretraining.npy.npz'
CHECKSUM1='b98e58647028eb33385c05b732191ba9'

FILE2='1Yb3qfwIS5KxfYgyzAhB_4uECG5Nl4Knf'
FILE2NAME='data_pretraining_dev_set.npy.npz'
CHECKSUM2='dc44a8c76e3d1cca7b6c4d05215ed467'

dl() {

        gdown $BASEURL$FILE1
        gdown $BASEURL$FILE2
}


check() {

        VAR1=$(md5sum ${FILE1NAME} | awk '{print $1}')

        # 1
        if [ "$CHECKSUM1" = "$VAR1" ]; then

            echo "Checksum OK: "$FILE1NAME

        else

            echo "Checksum NOT ok: "$FILE1NAME

        fi

        VAR2=$(md5sum ${FILE2NAME} | awk '{print $1}')

        # 2
        if [ "$CHECKSUM2" = "$VAR2" ]; then

            echo "Checksum OK: "$FILE2NAME

        else

            echo "Checksum NOT ok: "$FILE2NAME

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
