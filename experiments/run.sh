for dataset in  'ffhq1024' 'fashion-mnist' 'celeba'
do
    for quantizer in 'dense_lattice' 'sparse_lattice' 'vq' 'vq-ema'
    do
        python train.py dataset=$dataset quantizer=$quantizer
    done
done
