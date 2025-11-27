#!/bin/bash

DATASET=$1
RATE=$2
GPUID=$3

if [ "$DATASET" = "aapd" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 10 --batch 32  \
    --alpha 0.7 --epsilon 0.05  --theta 3\
    --rho $RATE

elif [ "$DATASET" = "movie" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 10 --batch 32  \
    --alpha 0.7 --epsilon 0.1  --theta 3 \
    --rho $RATE

elif [ "$DATASET" = "rcv" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 10 --batch 16  \
    --alpha 0.7 --epsilon 0.05  --theta 3 \
    --rho $RATE

elif [ "$DATASET" = "riedel" ]; then
    python main.py --gpuid $GPUID --dataset $DATASET \
    --epoch 30 --batch 256  \
    --alpha 0.6 --epsilon 0.15  --theta 3 \
    --rho $RATE
fi

