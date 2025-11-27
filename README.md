# LeD

 "Enhancing Multi-Label Text Classification under Label-Dependent Noise: A Label-Specific Denoising Framework" EMNLP 2024. 

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
