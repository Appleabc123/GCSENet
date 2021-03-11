# GCSENet

    GCSENet is a novel learning-based framework for miRNA-disease association identification via graph convolutional neural networks、convolutional neural networks、Squeeze-and-Excitation Networks（SENet）

it contains: "code" and "data"

    In the "code" folder, the four steps correspond to "1.Generate Feature by GCN", "2.Feature Process" , "3.Train" , "4.Test" respectively.
    In the "data" folder， it contains four part : Generate feature、process feature、CNN_SENet、Test

The user can apply the whole framework to their interested dataset or only use part of the framework.

#Dependencies
GCSENet was implemented with Python 3.6.4, with following packages installed:

    Matplotlib(3.1.1),     (https://pypi.org/project/matplotlib/)
    Networkx(2.5),       (https://pypi.org/project/networkx/)
    Tensorflow-gpu(1.4.0), (https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow-gpu/)
    Numpy(1.19.1),       (https://pypi.org/project/numpy/)
    Pandas(0.25.3),       (https://pypi.org/project/pandas/)
    Sklearn(0.20.3),       (https://pypi.org/project/sklearn/)
    Scipy(1.5.2),          ( https://pypi.org/project/scipy/)

In addition,CUDA 8.0 and cuDNN 6.0 have been used.

# Example

Step1. Get the feature vector (disease-gene, miRNA-gene)
    Set the data_path in main.py containing original data (d-d.csv, g-g.csv, d-g.csv,
    disease_name.csv, gene_name.csv)
    Run main.py to obtain disease-gene vector
    The result is saved in 
         ‘../data/process_feature’ folder, generating the ‘disease-gene.csv’ file 
    Similarly,
    Set the data_path in main.py containing original data (m-m.csv, g-g.csv, m-g.csv,
    miRNA_name.csv, gene_name.csv)
    Run main.py to obtain miRNA-gene vector
    The result is saved in
        ‘../data/process_feature’ folder, generating the ‘miRNA-gene.csv’ file 
        
        
Step2. Get the weighted feature (disease-miRNA) and label
    Run process_feature.py to obtain disease-miRNA vector
    The result is saved in
 	    ‘../data/CNN_SENet’ folder, generating two files with ‘disease-miRNA.csv’ and ‘label.csv’ 
        
        
Step3. Train the network
    Run CS_train.py to train the model
    
    
Step4. Test the benchmark set to get the AUROC, AUPR, Precision, Recall, F1-score
    Run CS_test.py to test the model


