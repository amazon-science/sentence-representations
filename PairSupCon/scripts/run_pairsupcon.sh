# Although we didn't use STS-B as the dev set for our EMNLP paper, we incorporate it here for your convience as we find STS-B providing good stopping indication when evaluated on different downstream tasks. 

python main.py \
    --resdir $path-to-store-your-results \
    --dev_set sts \
    --path_sts_data $path-to-the-senteval-datasets \
    --datapath $path-to-the-nli-dataset \
    --dataname nli_train_posneg \
    --text sentence \
    --pairsimi pairsimi \
    --num_classes 2  \
    --mode pairsupcon \
    --bert bertbase \
    --contrast_type HardNeg \
    --lr 5e-06 \
    --lr_scale 100 \
    --batch_size 1024 \
    --max_length 32 \
    --temperature 0.05 \
    --beta 1 \
    --epochs 3 \
    --max_iter 10000000 \
    --logging_step 500 \
    --seed 0 &





