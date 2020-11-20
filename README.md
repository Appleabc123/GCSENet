
# GCSENet

    GCSENet is a novel learning-based framework for miRNA-disease association identification via graph convolutional neural networks、convolutional neural networks、Squeeze-and-Excitation Networks（SENet）

it contains: "code" and "data"

    In the "code" folder, the four steps correspond to "1.Generate Feature by GCN", "2.Feature Process" , "3.Train" , "4.Test" respectively.
    In the "data" folder， it contains four part : Generate feature、process feature、CNN_SENet、Test
usage:

    If a researcher has a miRNA of interest or disease of interest in our dataset, they can put the disease and miRNA pairs in the "Test" folder to get the probability score

    If a researcher has a miRNA of interest or disease of interest out of our dataset, they can put the new disease\gene\miRNA into "Generate feature" folder to retrain the model to get the probability score

The user can apply the whole framework to their interested dataset or only use part of the framework.

GCSENet was implemented with Python 3.6.4.