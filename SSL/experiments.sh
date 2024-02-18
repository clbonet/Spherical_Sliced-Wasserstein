#!/bin/bash

cifar() {
    sbatch ./proxy.sh \
        --data_folder $WORK/ssw_ssl/data/ \
        --seed 0 \
        --feat_dim 3 \
        --batch_size 512 \
        --identifier _no_bias \
        $@
}

cifar --method ssw --unif_w 6 --num_projections 10
cifar --method ssw --unif_w 6 --num_projections 200
cifar --method hypersphere --unif_w 1 --align_w 1
cifar --method simclr
