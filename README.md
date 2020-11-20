# GCSENet

    GCSENet is a novel learning-based framework for miRNA-disease association identification via graph convolutional neural networks、convolutional neural networks、Squeeze-and-Excitation Networks（SENet）

it contains: "code" and "data"

    In the "code" folder, the four steps correspond to "1.Generate Feature by GCN", "2.Feature Process" , "3.Train" , "4.Test" respectively.
    In the "data" folder， it contains four part : Generate feature、process feature、CNN_SENet、Test
How to use:

    If a researcher has a miRNA of interest or disease of interest in our dataset, they can put the disease and miRNA pairs in the "Test" folder to get the probability score
    If a researcher has a miRNA of interest or disease of interest out of our dataset, they can put the new disease/gene/miRNA into "Generate feature" folder to retrain the model to get the probability score

The user can apply the whole framework to their interested dataset or only use part of the framework.

#Dependencies
GCSENet was implemented with Python 3.6.4, with following packages installed:

    numpy==1.19.1
    scipy==1.5.2
    networkx==2.5
    tensorflow-gpu==1.4.0

In addition,CUDA 8.0 and cuDNN 6.0 have been used.

#Example
If disease(heart failure) is our interest, we use several steps to get the final result :

Step1 : run "mian.py" in "1.Generate Feature by GCN" to get the feature vector

Step2 : run "process_feature.py" in "2.Feature Process" to get the weighted feature

Step3 : run "CS_train.py" in "3.Train" to train the network

Step4 : put the disease(heart failure) in the fold of ".data/Test/test.txt", run "CS_test.py" in "4.Test" to get the final predict result

