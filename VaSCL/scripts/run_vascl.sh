python main.py \
    --resdir ../results/vascl/  \
    --devset sts-b \
    --path_sts_data /home/ec2-user/code/release_code/DownstreamEval/SentEval/data \
    --datapath /home/ec2-user/efs/dejiao-explore/code/AugData/ \
    --dataname wiki1m_unique \
    --text1 text \
    --text2 text \
    --bert robertabase \
    --lr 5e-06 \
    --lr_scale 100 \
    --batch_size 256 \
    --epochs 5 \
    --max_iter 10000 \
    --logging_step 500 \
    --seed 0 &



#python main.py \
#    --resdir ../results/vascl/  \
#    --devset sts-b \
#    --path_sts_data PATH-TO-THE-SentEval-DATA \
#    --datapath PATH-TO-YOUR-PRETRAINING-DATA \
#    --dataname wiki1m_unique \
#    --text1 text \   # text1=text2, forwarding the same instance twice
#    --text2 text \
#    --bert robertabase \
#    --lr 5e-06 \
#    --lr_scale 100 \
#    --batch_size 1024 \
#    --epochs 5 \
#    --max_iter 100 \
#    --logging_step 500 \
#    --seed 0 &

