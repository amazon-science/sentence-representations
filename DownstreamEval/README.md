# Downstream tasks evaluation (EMNLP 2021) 

### Dependencies:
    python==3.7.10 
    transformers==4.8.1
    sentence-transformers==2.0.0
    numpy==1.19.2
    sklearn==0.23.2
    tensorboardX==2.3 


    
## To evaluate with your own PairSupCon checkpoints 
    python eval_cluster.py \
        --pretrained_dir /home/ec2-user/efs/dejiao-explore/experiments/train/pairsupcon/ \ # pretrained ckpt directory
        --respath /home/ec2-user/efs/dejiao-explore/exprres/test_eval/ \  # results directory
        --pdataname nli_train_posneg \ #pretrained data name
        --eval_instance local \ # use local gpu/cpu instance or sagemaker
        --mode pairsupcon \
        --bert distilbert \
        --contrast_type HardNeg \
        --temperature 0.05 \
        --beta 1 \
        --lr 5e-06 \
        --lr_scale 100 \
        --p_batchsize 1024 \ #pretrained_batchsize
        --pretrain_epoch 3 \
        --pseed 0 \ #pretrained_seed
        --seed 0 \
        --device_id 0 &



## To evaluate your own non-PairSupCon checkpoints
    you need modify the following 
        1. "get_checkpoint" function in ./configs/configure
        2. "get_embeddings" function in ./clustering/clustering_eval.py
        3. "batcher" function in ./semsimilarity/sts_eval.py



