# Influenza-virus
A high-throughput viral RNA replication analysis (HT-VRNArep) framework, which combines deep mutagenesis, high-throughput sequencing and deep learning models to systematically investigate the replication activity landscape of the influenza A virus RNA 3'-terminus.
## Contents
[1.Introduction](#Introduction)    
[2.Environment Setup](#Environment-Setup)    
[3.Ngs_analysis](#Ngs_analysis)  
[4.Train model](#Train-model)    
[5.Interpretation](#Interpretation)    
## Introduction
A high-throughput viral RNA replication analysis (HT-VRNArep) framework, which combines deep mutagenesis, high-throughput sequencing and deep learning models to systematically investigate the replication activity landscape of the influenza A virus RNA 3'-terminus.
## Environment Setup
### Env Requirements:
MAC OS, Linux or Windows.   
Python 3.9.19  
Tensorflow 2.16.1 + keras 3.3.3  
## Ngs_analysis
```
python ngs_analysis.py process \
  --virus-name <str> \
  --type <str> \
  --date <str> \
  --forward_fastq forward.fq \
  --reverse_fastq reverse.fq \
  --upstream <str> \
  --downstream <str> \
  --wt <str>
```
```
python ngs_analysis.py score \
  --dna dna_results.csv \
  --rna rna_results.csv \
  --output merged_score.csv \
  --split \
  --test-size <float>
```
## Train model
```
python train.py <model> train  
  --virus-name=<str> 
  --date=<str> 
  [--num-train=<int>] 
  [--save] 
  [--epochs=<int>] 
  [--input-file=<str>] 
  [--output-file=<str>] 
  [--strand-specific] 
  [--batch=<int>]
  [--learning-rate=<float>]
```
## Interpretation
  REPLICATOR/code/deepcnn_interpretation.ipynb  
  REPLICATOR/code/epistasis.ipynb  
