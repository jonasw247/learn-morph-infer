#!/bin/bash

python3 main.py --is_3d=True --dataset 'Dataset' --outputparams=7 --res_x=128 --res_y=128 --res_z=128 --filters=32 --log_step=50 --num-conv=3 --batch_size=12 --num_worker=8 --log_ep=2 --test_ep=2 --use_curl False  --arch 'alternative' --phys_loss False --is_train True --gpu_id="1" --max_epoch=10 --lr_max=0.0003 --beta1=0.9 --tag 'inverse_test_thresholded_increasedlr' --valid_dataset_dir '/mnt/Drive2/ivan/data/valid' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --random_seed=47393

