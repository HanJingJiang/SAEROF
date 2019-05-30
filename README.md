 SAEROF

Copyright (C) 2019 Han-Jing Jiang(jianghanjing17@mails.ucas.ac.cn),Yu-An Huang(yu-an.huang@connect.polyu.hk),Zhu-Hong You(zhuhongyou@xjb.ac.cn)

Computational drug repositioning
---
We here propose a drug repositioning computational method combining the techniques of Sparse auto-encoder and rotation forest (SAEROF) which is able to learn new features effectively representing drug-disease associations via its hidden layers.Gaussian interaction profile kernel similarity, drug structure similarity and disease semantic similarity were extracted respectively. Based on the feature representation, a rotation forest classifier based on sparse auto-encoder is introduced to predict drug-disease interaction. 

Dataset
---
1.CdiseaseSimilarity store disease similarity matrix of Cdataset
2.diseaseSimilarity store disease similarity matrix of Fdataset
3.Drug -disease-whole and c-drug-disease-whole store known drug-disease associations of Cdataset and Fataset.

code
---
1.Feature.py:Function to generate the total characteristics
2.sparse auto-encoder.py:The features are obtained by sparse auto-encoder
3.Rotation forest.py:Sparse auto-encoder algorithm
4.Rof.py:predict potential indications for drugs

All files of Dataset and Code should be stored in the same folder to run SAEROF.
