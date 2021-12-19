# Downstream tasks evaluation (EMNLP 2021) 

### Dependencies:
    python==3.7.10 
    transformers==4.8.1
    sentence-transformers==2.0.0
    numpy==1.19.2
    sklearn==0.23.2
    tensorboardX==2.3 
    pandas


    
## To evaluate our PairSupCon Checkpoints 

### STS
    1. dataset: download the dataset by runing the following: 
              ./SentEval/data/downstream/download_dataset.sh

    2. bash run_sts.sh
   
   
### Clustering 

    1. download the clustering datasets from https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data, 
       1a. convert each dataset to csv files with column names "text" (for text samples) and "label" (for cluster labels)
       1b. store it in "your-path-to-cluster-data"

    2. bash run_clustering.sh






