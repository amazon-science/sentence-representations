python main.py \
   --resdir ../results/vascl/  \
   --devset sts-b \
   --path_sts_data PATH-TO-THE-SentEval-DATA \
   --datapath PATH-TO-YOUR-PRETRAINING-DATA \
   --dataname wiki1m_unique \
   --text1 text \   # text1=text2, forwarding the same instance twice
   --text2 text \
   --bert robertabase \
   --lr 5e-06 \
   --lr_scale 100000 \
   --batch_size 1024 \
   --epochs 5 \
   --max_iter 100 \
   --logging_step 500 \
   --seed 0 &

