for seed in 32 432 2
    dataset in 'ffhq1024' 'celeba' 'fashion-mnist'
    do
        python train.py quantizer='dense_lattice' dataset=$dataset general.seed=$seed  general.device=0
        python train.py quantizer='sparse_lattice_init_b' dataset=$dataset general.seed=$seed  general.device=0
        python train.py quantizer='sparse_lattice_no_init_b' dataset=$dataset general.seed=$seed  general.device=0

        for quantizer in 'vq' 'vq-ema'
        do
            python train.py quantizer=$quantizer dataset=$dataset general.seed=$seed  general.device=0
        done
    done