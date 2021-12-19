
# python eval_sts.py \
#     --path_to_sts_data /home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/ \
#     --pretrained_model PairSupCon \
#     --bert pairsupcon-base \
#     --respath /home/ec2-user/efs/dejiao-explore/experiments/evaluation/PSC_Baseline/ 
    

python eval_sts.py \
    --path_to_sts_data /home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/ \
    --pretrained_model PairSupCon \
    --bert pairsupcon-large \
    --respath /home/ec2-user/efs/dejiao-explore/experiments/evaluation/PSC_Baseline/ 