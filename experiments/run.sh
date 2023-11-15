for dataset in 'fashion-mnist' 'celeba' 'ffhq1024'
do
    for quantizer in 'dense_lattice' 'sparse_lattice' 'vq' 'vq-ema'
    do
        python evaluate_checkpoint.py dataset=$dataset quantizer=$quantizer
    done
done
