#!/bin/bash
python3 main.py --isnewsave --outputmode 3 --gpu 3 --purpose norm-necr-normpet-n4-128 --batch_size 12 --num_workers 5 --starttrain 0 --endtrain 79992 --startval 80000 --endval 87992 --dropoutrate 0.0 --lr 0.00002 --lr_scheduler_rate 0.999998 --weight_decay_sgd 0.01
