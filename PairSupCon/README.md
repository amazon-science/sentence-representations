# Pairwise Supervised Contrastive Learning of Sentence Representations (EMNLP 2021) 
Dejiao Zhang, Shang-Wen Li, Wei Xiao, Henghui Zhu,
Ramesh Nallapati, Andrew Arnold, and Bing Xiang. 

This repository contains the code for our paper [Pairwise Supervised Contrastive Learning of Sentence Representations](https://aclanthology.org/2021.emnlp-main.467/).


### Dependencies:
    python==3.7.10 
    transformers==4.8.1
    sentence-transformers==2.0.0
    numpy==1.19.2
    sklearn==0.23.2
    tensorboardX==2.3 


## To run the code:
    bash ./scripts/run_pairsupcon.sh
    
    

## Run PairSupCon with your specific parameters

    python main.py \
        --resdir /home/ec2-user/efs/dejiao-explore/experiments/train/pairsupcon/ \ # results directory
        --use_stsb_dev \  # use STS-B as dev set or not, default False
        --path_sts_data /home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/ \ # STS-B data path
        --datapath s3://dejiao-experiment-meetingsum/datasets/NLI/ \ # NLI training data path 
        --dataname nli_train_posneg \
        --text sentence \  #column name for the NLI sentences
        --pairsimi pairsimi \ #column name for the NLI pairwise labels
        --num_classes 2  \ # focuses on Entailment and Contradiction classification only
        --mode pairsupcon \ 
        --bert distilbert \
        --contrast_type HardNeg \
        --use_stsb_dev \
        --lr 5e-06 \
        --lr_scale 100 \
        --batch_size 1024 \ #use a smaller batch size with single gpu
        --max_length 32 \
        --temperature 0.05 \
        --beta 1 \
        --epochs 1 \
        --max_iter 10 \
        --logging_step 250 \
        --seed 0 &



## Citation:
    @inproceedings{zhang-etal-2021-pairwise,
    title = "Pairwise Supervised Contrastive Learning of Sentence Representations",
    author = "Zhang, Dejiao  and
      Li, Shang-Wen  and
      Xiao, Wei  and
      Zhu, Henghui  and
      Nallapati, Ramesh  and
      Arnold, Andrew O.  and
      Xiang, Bing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.467",
    pages = "5786--5798",
    abstract = "Many recent successes in sentence representation learning have been achieved by simply fine-tuning on the Natural Language Inference (NLI) datasets with triplet loss or siamese loss. Nevertheless, they share a common weakness: sentences in a contradiction pair are not necessarily from different semantic categories. Therefore, optimizing the semantic entailment and contradiction reasoning objective alone is inadequate to capture the high-level semantic structure. The drawback is compounded by the fact that the vanilla siamese or triplet losses only learn from individual sentence pairs or triplets, which often suffer from bad local optima. In this paper, we propose PairSupCon, an instance discrimination based approach aiming to bridge semantic entailment and contradiction understanding with high-level categorical concept encoding. We evaluate PairSupCon on various downstream tasks that involve understanding sentence semantics at different granularities. We outperform the previous state-of-the-art method with 10{\%}{--}13{\%} averaged improvement on eight clustering tasks, and 5{\%}{--}6{\%} averaged improvement on seven semantic textual similarity (STS) tasks.",
}
