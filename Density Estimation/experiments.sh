#!/bin/bash

python train_density_earth_iter.py --loss "ssw" --dataset "fire" --n_epochs 20001 --n_try 5
# python train_density_earth_iter.py --loss "ssw" --dataset "flood" --n_epochs 20001 --n_try 5
# python train_density_earth_iter.py --loss "ssw" --dataset "quakes_all" --n_epochs 20001 --n_try 5
