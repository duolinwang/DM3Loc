# DM3Loc

DM3Loc is a novel Deep-learning framework with multi-head self-attention for multi-label mRNA subcellular localization prediction and analyses, which provide prediciton for six subcellular compartments, including nucleus, exosome, cytoplasm, ribosome, membrane, and endoplasmic reticulum.

##### Installation

  - Installation has been tested in Ubuntu 16.04.5 LST and Mac OS HighSierra with python 3.5.2: 
You can install the dependent packages by the following commands:
    ```sh
    sudo apt-get install -y python3.5 python3-pip
    python3 -m pip install numpy 
    python3 -m pip install scipy
    python3 -m pip install scikit-learn
    python3 -m pip install pillow
    python3 -m pip install h5py
    python3 -m pip install keras==2.2.4
    python3 -m pip install tensorflow==1.12.0 (or install the GPU supported tensorflow by pip3 install tensorflow-gpu==1.12.0 refer to https://www.tensorflow.org/install/ for instructions)
    ```
    Download the stand-alone tool by:
    ```sh
    git clone https://github.com/duolinwang/DM3Loc
    ```
##### Running on GPU or CPU
>If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions. 
CPU is only suitable for prediction not training. 

#### For general users who want to perform mRNA subcellular localization prediction by our provided model :
can directly call Multihead_predict.py 
```sh
python3 Multihead_predict.py --dataset [custom mRNA data in FASTA format] --outputpath [custom specified output folder for the prediction results]
```
For details of the parameters, use the -h or --help parameter.
For example, to predict for Test_fold0.fasta in the testdsata folder and output the results to the folder ./Results/5foldresult/, run the following command:
```sh
python3 Multihead_predict.py --dataset "./testdata/test.fasta" --outputpath ./Results/
```
The result folder contains: 1) prediction_results.txt, the prediction result file. 2) attention_weights.txt, attention weight file, it contains attention weight vector per mRNA seuqnece.
#### For advanced users like to perform training by using their own data
Call Multihead_train.py. Multihead_train.py contains the process of spliting user's custom seuqneces into 5-fold cross-validation data and training 5 folds of models. The final model can be an ensemble of these 5 folds of models. To train customized models, users can run the following commands and replace with their own data and parameters.
```sh
python3 Multihead_train.py --normalizeatt --classweight --dataset [custom training data in FASTA format] --epochs 500 --message [output custom model keywords]
```
The training data should be in the FASTA format. The labels should be presented by onehot encoding and in the first place of the mRNA sequence description, seperated with other parts by a comma ',', as in the following example: 
```sh
>010000,ACCNUM:NM_001507,Gene_ID:2862,Gene_Name:MLNR
ATGGGCAGCCCCTGGAACGGCAGCGACGGC.....................
```
Parameters --normalizeatt --classweight were required to train the model in the paper. For details of other parameters, use the -h or --help parameter.
##### Examples of commands used to train the models for 5-fold cross-validation:
 ```sh
python3 Multihead_train.py --normalizeatt --foldnum 5 --classweight --dataset ./testdata/modified_multilabel_seq_nonredundent.fasta --epochs 500 --message cnn64_smooth_l1
```
Note that the final model used in DM3Loc webserver was an ensemble model trained by 8-folds cross-validation data, to obtain that model, users should set the parameter --foldnum 8 and use the whole dataset "modified_multilabel_seq", which contains the redundant sequences, to train the model. 

### The DM3Loc webserver can be accessed at http://dm3loc.lin-group.cn/
