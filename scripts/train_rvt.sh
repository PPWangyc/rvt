#!/bin/bash
LOGDIR=logs
IMGDIR=/path/to/Data/breathe
OUTDIR=/path/to/rvt/output/output_rvt.json
DATACONFIG=/path/to/rvt/configs/data_rvt.json
RESUME=/path/to/rvt/checkpoint/model_set_1.pth
EPOCHS=40
GPU=0
LR=0.000005
seed=2

cd ..
for i in {1..24}
do  
    python rvt.py --img_path $IMGDIR \
                --out_dir $OUTDIR \
                --lr $LR \
                --epochs $EPOCHS \
                --optimizer sgd \
                --resume $RESUME \
                --batch_size 1 \
                --gpu $GPU \
                --seed $seed \
                --config_data $DATACONFIG \
    > $LOGDIR/train.log
    cd utils
    python k_cross.py --config_data $DATACONFIG
    cd .. 
done
cd scripts

echo "finished"