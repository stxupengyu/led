# LeD

Official code for our paper. 

## Requirements: 
* python==3.9.19 
* numpy==1.26.4   
* scipy==1.13.0  
* scikit-learn==1.4.2
* torch==2.3.0
* gensim==4.1.2
* nltk==3.8.1
* tqdm==4.66.2
* tokenizers==0.19.1 
* transformers==4.41.2


## Datasets
LeD uses the same datasets with [nEM](https://ieeexplore.ieee.org/document/9975199).
The original datasets are available from the following links. 

* [MOVIE](https://github.com/davidsbatista/text-classification)
* [AAPD](https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD/data)
* [RCV1](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)
* [Riedel](https://github.com/AlbertChen1991/nEM)

The processed dataset are available from the following links.
please download them from the following links.
Due to copyright issues, we can't directly release the RCV1 datasets used in our experiments. Instead, we provide the link to the data sources (require license).

* [MOVIE](https://www.dropbox.com/scl/fi/mpodhwyxj7y25i9pj25en/MOVIE.zip?rlkey=w2rgb59p6mt4tcdtla5vtguma&e=1&st=0rs70q5i&dl=0)
* [AAPD](https://www.dropbox.com/scl/fi/veorgip9rawzocd6z51tz/AAPD.zip?rlkey=4gvuxmetaox2e46ie9oxxvt36&e=1&st=l99z6yze&dl=0)
* [RCV1](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) 
* [Riedel](https://drive.google.com/drive/folders/1u2HVCYoJcV5SiFcmrP0yEIn5E5tJ6Mbg)


## Experiments

### Data Path
Please confirm the corresponding configuration file. Make sure the data path parameters (data_dir, dataset and etc.) are right in:   
```bash
main.py
```

### Train and Evaluate
```bash
bash script/run.sh <dataset> <noise_rate> <gpu_id> 
```
options:  
`<dataset>`: aapd, rcv, movie, riedel.  
`<noise_rate>`: 0.2, 0.4, 0.6