for dataset in 'ffhq1024' 'celeba' 'fashion-mnist'
    for seed in 32 432 2
    do
        python train.py quantizer='ll-vq-vae' dataset=$dataset general.seed=$seed  general.device=0 quantizer.args.initialize_embedding_b=False quantizer.args.sparsity_cost=-1.0 quantizer.count_uniques_bool=False

        for quantizer in 'vq-vae' 'vq-vae-ema'
        do
            python train.py quantizer=$quantizer dataset=$dataset general.seed=$seed  general.device=0
        done

        for initialize_embedding_b in True False
        do
            python train.py quantizer='ll-vq-vae' dataset=$dataset general.seed=$seed  general.device=0 quantizer.args.initialize_embedding_b=$initialize_embedding_b quantizer.args.sparsity_cost=1.0 quantizer.count_uniques_bool=True
        done
    done