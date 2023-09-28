for dataset in 'ffhq1024' 'celeba' 'fashion-mnist'
    for seed in 32 432 2
    do
        python train.py model='ll-vq-vae' dataset=$dataset general.seed=$seed  general.device=0 model.args.initialize_embedding_b=False model.args.sparsity_cost=-1.0 model.count_uniques_bool=False

        for model in 'vq-vae' 'vq-vae-ema'
        do
            python train.py model=$model dataset=$dataset general.seed=$seed  general.device=0
        done

        for initialize_embedding_b in True False
        do
            python train.py model='ll-vq-vae' dataset=$dataset general.seed=$seed  general.device=0 model.args.initialize_embedding_b=$initialize_embedding_b model.args.sparsity_cost=1.0 model.count_uniques_bool=True
        done
    done