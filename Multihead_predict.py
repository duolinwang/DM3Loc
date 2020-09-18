import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
basedir='./'
sys.path.append(basedir)
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
set_session(session=sess)

from multihead_attention_model import *
from Genedata import Gene_data
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc


encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))


gene_ids = None
batch_size = 100 #try 200 later! too slow!
nb_classes = 6

def batch(iterable1,iterable2, batch_size=1):
    l = len(iterable1)
    for ndx in range(0, l, batch_size):
        yield [iterable1[ndx:min(ndx + batch_size, l)],iterable2[ndx:min(ndx + batch_size, l)]]

def get_label(label):
    #assert (len(dist) == 4)
    return [int(x) for x in label]

def label_dist(dist):
    #assert (len(dist) == 4)
    return [int(x) for x in dist]

def maxpooling_mask(input_mask,pool_length=3):
    #input_mask is [N,length]
    max_index = int(input_mask.shape[1]/pool_length)-1
    max_all=np.zeros([input_mask.shape[0],int(input_mask.shape[1]/pool_length)])
    for i in range(len(input_mask)):
        index=0
        for j in range(0,len(input_mask[i]),pool_length):
            if index<=max_index:
                max_all[i,index] = np.max(input_mask[i,j:(j+pool_length)])
                index+=1
    
    return max_all

def convertattweight2inputAxis(att_weight,att_mask,length =8000,pooling_size=8):
    '''
    att_weight is a list with length 2666 need to be converted to 8000
    '''
    seqWeight = []
    for pos in range(len(att_weight)):
        seqWeight.extend(np.repeat(att_weight[pos],pooling_size) )
    
    #pad_value = seqWeight[-1]
    #print("padvalue is:"+str(pad_value))
    #print("attweight is :")
    #print(len(att_weight))
    padlength = length-len(att_weight)*pooling_size
    if padlength>0:
        seq = np.pad(np.asarray(seqWeight),([0,padlength]),'constant',constant_values =att_weight.min())
    else:
        seq = seqWeight[:length]
    
    seq=seq*att_mask
    #print("lenght of real is")
    #print(np.sum(att_mask))
    return seq[np.where(att_mask!=0)]


def att_normolization(x,method='zscore',relu=False):
    '''
    x [sample, head, length]
    '''
    from scipy import stats
    modenumthres=10
    y = x.copy()            
    if method == 'zscore':
      y = stats.zscore(y,axis=-1)
      y = np.nan_to_num(y,0)
      return y


def preprocess_data(left, right,dataset,padmod='center',pooling_size=3,evaluate=False):
    gene_data = Gene_data.load_sequence(dataset, left, right,predict=True)
    geneids = [gene.id for gene in gene_data]
    maxpoolingmax = int((left+right)/pooling_size)
    #pad center
    X_left = [[seq_encoding_keys.index(c.upper()) for c in gene.seqleft] for gene in gene_data]
    X_right = [[seq_encoding_keys.index(c.upper()) for c in gene.seqright] for gene in gene_data]
    genelength = [int(gene.length) for gene in gene_data]
    if padmod =='center':
        mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
        mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
        mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
        mask_label_pooling=maxpooling_mask(mask_label,pool_length=pooling_size)
        X_left = pad_sequences(X_left,maxlen=left,
                           dtype=np.int8, value=seq_encoding_keys.index('UNK'),padding='post')  #padding after sequence
        
        X_right = pad_sequences(X_right,maxlen=right,
                      dtype=np.int8, value=seq_encoding_keys.index('UNK'),padding='pre')# padding before sequence
        
        X = np.concatenate([X_left,X_right],axis = -1)
    else:
       #merge left and right and padding after sequence
       Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
       X = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=seq_encoding_keys.index('UNK'),padding='post')
       mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
       #Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
       mask_label_pooling=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
    
    if evaluate:
        y = np.array([label_dist(gene.label) for gene in gene_data])
        return X,y,mask_label,mask_label_pooling,geneids,genelength
    else:
        return X,mask_label,mask_label_pooling,geneids,genelength


def run_model(lower_bound, upper_bound, dataset, **kwargs):
    '''load data into the playground'''
    
    classlist = ["Nucleus","Exosome","Cytoplasm","Ribosome","Membrane","Endoplasmic reticulum"]
    import pickle
    max_len = kwargs['left']+kwargs['right']
    if kwargs['evaluate']:
        X, y, mask_label,mask_label_pooling,geneids,genelength = preprocess_data(kwargs['left'],kwargs['right'], dataset,kwargs['padmod'],kwargs['pooling_size'],True)
    
    else:
        X, mask_label,mask_label_pooling,geneids,genelength = preprocess_data(kwargs['left'],kwargs['right'], dataset,kwargs['padmod'],kwargs['pooling_size'],False)
    
    avg_predicted_y=np.zeros([X.shape[0],nb_classes])
    att_matrix1_all=np.zeros([kwargs['foldnum'],X.shape[0],kwargs['headnum'],int(max_len/kwargs['pooling_size'])])
    att_matrix2_all=np.zeros([kwargs['foldnum'],X.shape[0],kwargs['headnum'],int(max_len/kwargs['pooling_size'])])
    att_matrix3_all=np.zeros([kwargs['foldnum'],X.shape[0],kwargs['headnum'],int(max_len/kwargs['pooling_size'])])
    #average_att_123=np.zeros([X.shape[0],int(max_len/kwargs['pooling_size'])])
    
    i=0
    model = multihead_attention(max_len, nb_classes, OUTPATH, kfold_index=i)  # initialize here load weights after model initialization
    model.build_model_multihead_attention_multiscaleCNN4_covermore(
                                             dim_attention=kwargs['dim_attention'],
                                             headnum=kwargs['headnum'],
                                             embedding_vec=seq_encoding_vectors,
                                             nb_filters=kwargs['nb_filters'],
                                             filters_length1=kwargs['filters_length1'],
                                             filters_length2=kwargs['filters_length2'],
                                             filters_length3=kwargs['filters_length3'],
                                             pooling_size=kwargs['pooling_size'],
                                             drop_input=kwargs['drop_input'],
                                             drop_cnn=kwargs['drop_cnn'],
                                             drop_flat=kwargs['drop_flat'],
                                             W1_regularizer=kwargs['W1_regularizer'],
                                             W2_regularizer=kwargs['W2_regularizer'],
                                             Att_regularizer_weight=kwargs['Att_regularizer_weight'],
                                             fc_dim = kwargs['fc_dim'],
                                             fcnum = kwargs['fcnum'],
                                             posembed=kwargs['posembed'],
                                             pos_dmodel=kwargs['pos_dmodel'],
                                             pos_nwaves = kwargs['pos_nwaves'],
                                             posmod = kwargs['posmod'],
                                             regularfun = kwargs['regularfun'],
                                             huber_delta=kwargs['huber_delta'],
                                             activation = kwargs['activation'],
                                             add_avgpooling = kwargs['add_avgpooling'],
                                             poolingmod = kwargs['poolingmod'],
                                             load_weights=True,
                                             weight_dir = kwargs['weights_dir']+str(i)+".h5",
                                             normalizeatt=kwargs['normalizeatt'],
                                             attmod=kwargs['attmod'],
                                             sharp_beta=kwargs['sharp_beta']
                                            )
    
    batch_size = 100
    totalindex = int(np.ceil(float(len(X)/batch_size)))
    totalnum=totalindex*kwargs['foldnum']
    batch_generator = batch(X,mask_label_pooling,batch_size)
    for index in range(totalindex):
        websiteoutput = open(OUTPATH+"/prediction_predicted_num.txt",'w')
        prossratio = round(float(index)/(totalindex)*100,2);
        websiteoutput.write("Processed:"+str(prossratio)+"\n")
        websiteoutput.close()
        batch_data = next(batch_generator)
        for i in range(kwargs['foldnum']):
            model.model.load_weights(kwargs['weights_dir']+str(i)+".h5")
            pred_y = model.model.predict([batch_data[0],batch_data[1].reshape(-1,batch_data[1].shape[1],1)])
            avg_predicted_y[index*batch_size:index*batch_size+len(batch_data[0])]+=pred_y
            #add attention
            att_matrix1,att_matrix2,att_matrix3 = model.get_attention_multiscale_batch(batch_data[0],batch_data[1])  #(sample,head,length)
            att_matrix1_all[i,index*batch_size:index*batch_size+len(batch_data[0])]= att_matrix1
            att_matrix2_all[i,index*batch_size:index*batch_size+len(batch_data[0])]= att_matrix2
            att_matrix3_all[i,index*batch_size:index*batch_size+len(batch_data[0])]= att_matrix3
            #group_att_avg1= np.average(att_normolization(att_matrix1,relu=True),axis=1) #(sample,length)
            #group_att_avg2= np.average(att_normolization(att_matrix2,relu = True),axis=1)
            #group_att_avg3= np.average(att_normolization(att_matrix3,relu = True),axis=1)
            ##group_att_avg1= np.average(att_matrix1,axis=1) #(sample,length)
            ##group_att_avg2= np.average(att_matrix2,axis=1)
            ##group_att_avg3= np.average(att_matrix3,axis=1)
            #average_att_123[index*batch_size:index*batch_size+len(batch_data[0])]+=np.max(np.concatenate([np.expand_dims(group_att_avg1,2),np.expand_dims(group_att_avg2,2),np.expand_dims(group_att_avg3,2)],axis=-1),axis=-1)
    #(sample,length)
    ################
    K.clear_session()
    
    #average_att_123=average_att_123/kwargs['foldnum']
    avg_predicted_y = avg_predicted_y/kwargs['foldnum']
    #print("shape of avg_predicted_y is "+str(avg_predicted_y.shape))
    outfile=open(OUTPATH+"/prediction_results.txt",'w')
    outfile.write("ID\tNucleus,Exosome,Cytoplasm,Ribosome,Membrane,Endoplasmic reticulum,Predicted Subcellular Location (cutoffs:Nu=0.66,Ex=0.98,Cy=0.55,Ri=0.32,Me=0.24,En=0.21)\n")
    defaultcutoff=[0.66,0.98,0.55,0.32,0.24,0.21]
    for i in range(len(geneids)):
        outfile.write(geneids[i]+"\t")
        label = []
        for c in range(nb_classes-1):
            outfile.write("%0.3f"%(avg_predicted_y[i,c])+",")
            if avg_predicted_y[i,c]>defaultcutoff[c]:
                label.append(classlist[c])
        
        outfile.write("%0.3f"%(avg_predicted_y[i,nb_classes-1]))
        if avg_predicted_y[i,nb_classes-1]>defaultcutoff[nb_classes-1]:
                label.append(classlist[nb_classes-1])
        
        if len(label)==0:
            label = ["None"]
        
        outfile.write("\t"+",".join(label)+"\n")
    
    outfile.close()
    #add attention
    outfile=open(OUTPATH+"/attention_weights.txt",'w')
    for i in range(len(geneids)):
         outfile.write(geneids[i]+"\t")
         extraLength = int(np.sum(mask_label[i])/kwargs['pooling_size'])
         average_att_123=np.zeros(extraLength)
         for fold in range(kwargs['foldnum']):
            group_att_avg1= np.average(att_normolization(att_matrix1_all[fold][i,:,:extraLength],relu=True),axis=0) #(length)
            group_att_avg2= np.average(att_normolization(att_matrix2_all[fold][i,:,:extraLength],relu = True),axis=0)
            group_att_avg3= np.average(att_normolization(att_matrix3_all[fold][i,:,:extraLength],relu = True),axis=0)
            #print("shape of group_att_avg1 is ")
            #print(group_att_avg1.shape)
            average_att_123+=np.max(np.concatenate([np.expand_dims(group_att_avg1,1),np.expand_dims(group_att_avg2,1),np.expand_dims(group_att_avg3,1)],axis=-1),axis=-1)
         
         average_att_123=average_att_123/kwargs['foldnum']
         #print("length is "+str(len(convertattweight2inputAxis(average_att_123[i],mask_label[i],max_len,kwargs['pooling_size']))))
         #print(str(np.sum(mask_label[i])))
         #attweights = convertattweight2inputAxis(average_att_123[i][:extraLength],mask_label[i],max_len,kwargs['pooling_size'])
         attweights = convertattweight2inputAxis(average_att_123[:extraLength],mask_label[i],max_len,kwargs['pooling_size'])
         #if i<10:
         #   print(genelength[i])
         
         if genelength[i]>max_len:
            att_left = attweights[:kwargs['left']]
            att_right = attweights[kwargs['left']:]
            padlength = genelength[i]-max_len
            attweights = np.concatenate([att_left,np.zeros(padlength),att_right])
         
         outfile.write(','.join([str(x) for x in attweights]))
         outfile.write("\n")
    
    outfile.close()
    ##############
    if kwargs['evaluate']:
         roc_auc = dict()
         mcc_dict = dict()
         average_precision = dict()
         #for ROC plot
         fpr = dict()
         tpr = dict()
         precision = dict()
         recall = dict()
         for i in range(nb_classes):#calculate one by one
             average_precision[i+1] = average_precision_score(y[:, i], avg_predicted_y[:, i])
             roc_auc[i+1] = roc_auc_score(y[:,i], avg_predicted_y[:,i])
             mcc_dict[i+1] = matthews_corrcoef(y[:,i],[1 if x>0.5 else 0 for x in avg_predicted_y[:,i]])
             fpr[i],tpr[i],_ = roc_curve(y[:,i], avg_predicted_y[:,i])
             precision[i],recall[i],_ =  precision_recall_curve(y[:, i], avg_predicted_y[:, i])
         
         
         fpr['micro'],tpr['micro'],_ = roc_curve(y.ravel(), avg_predicted_y.ravel())
         precision['micro'],recall['micro'],_ =  precision_recall_curve(y.ravel(), avg_predicted_y.ravel())
         average_precision["micro"] = average_precision_score(y, avg_predicted_y,average="micro")
         roc_auc["micro"] = roc_auc_score(y,avg_predicted_y,average="micro")
         roc_list = [roc_auc[x+1] for x in range(nclass)]
         roc_list.append(roc_auc['micro'])
         pr_list = [average_precision[x+1] for x in range(nclass)]
         pr_list.append(average_precision['micro'])
         mcc_list = [mcc_dict[x+1] for x in range(nclass)]
         np.savetxt(OUTPATH + '/evaluation_roc_average_presicion.txt', np.array(roc_list+pr_list+mcc_list), delimiter=',')
         
         picklefile = open(OUTPATH + '/evaluation_plot','wb')
         pickle.dump((fpr,tpr,precision,recall),picklefile)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Model parameters'''
    parser.add_argument('--lower_bound', type=int, default=3000, help='set lower bound on sample sequence length')
    parser.add_argument('--upper_bound', type=int, default=3000, help='set upper bound on sample sequence length')
    parser.add_argument('--left', type=int, default=4000, help='set left on sample sequence length')
    parser.add_argument('--right', type=int, default=4000, help='set left on sample sequence length')
    parser.add_argument('--dim_attention', type=int, default=80, help='dim_attention')
    parser.add_argument('--headnum', type=int, default=5, help='number of multiheads') #select one from 3
    parser.add_argument('--drop_input', type=float, default=0.06, help='drop_input')
    parser.add_argument('--drop_cnn', type=float, default=0.25, help='drop_cnn')
    parser.add_argument('--drop_flat', type=float, default=0.26, help='drop_flat')
    parser.add_argument('--W1_regularizer', type=float, default=0.001, help='W_regularizer')
    parser.add_argument('--W2_regularizer', type=float, default=0.001, help='W_regularizer')
    parser.add_argument('--Att_regularizer_weight', type=float, default=0.001, help='Att_regularizer_weight')
    parser.add_argument('--dataset', type=str, default='../../mRNAsubloci_train.fasta', help='input sequence data')
    parser.add_argument('--epochs', type=int, default=500, help='')
    parser.add_argument('--nb_filters', type=int, default=64, help='number of CNN filters') #select one from 3
    parser.add_argument('--filters_length', type=int, default=10, help='size of CNN filters') #select one from 3
    parser.add_argument('--filters_length1', type=int, default=9, help='size of CNN filters') #select one from 3
    parser.add_argument('--filters_length2', type=int, default=20, help='size of CNN filters') #select one from 3
    parser.add_argument('--filters_length3', type=int, default=49, help='size of CNN filters') #select one from 3
    parser.add_argument('--pooling_size', type=int, default=8, help='pooling_size') #select one from 3
    parser.add_argument('--fc_dim', type=int, default=100, help='fc_dim')
    parser.add_argument('--fcnum', type=int, default=1, help='fcnum')
    parser.add_argument("--outputpath", type=str, default="", help="append to the dir name")
    parser.add_argument("--weights_dir", type=str, default="./model/weights_fold_",
                        help="Must specificy pretrained weights dir for prediction") 
    
    parser.add_argument("--foldnum", type=int, default=8,help="foldnum") 
    
    parser.add_argument("--evaluate",action="store_true",help="whether to evaluate the test result, if set, labels must be provided")
    parser.add_argument("--posembed", action="store_true",help="use posembed")
    parser.add_argument("--pos_dmodel", type=int,default=40,help="pos_dmodel")
    parser.add_argument("--pos_nwaves", type=int,default=20,help="pos_nwaves")
    parser.add_argument("--posmod", type=str,default='concat',help="posmod")
    parser.add_argument("--regularfun",type=int,default=1,help = 'regularfun for l1 or l2 3 for huber_loss')
    parser.add_argument("--huber_delta",type=float,default=1.0,help = 'huber_delta')
    
    parser.add_argument("--activation",type=str,default='gelu',help = 'activation')
    parser.add_argument("--add_avgpooling", action="store_true",help="add_avgpooling")
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--classweight', action="store_true", help='classweight')
    parser.add_argument('--poolingmod',type=int,default=1,help = '1:maxpooling 2:avgpooling')
    parser.add_argument("--padmod", type=str,default='after',help="padmod: center, after")
    parser.add_argument("--normalizeatt", action="store_true",help="normalizeatt")
    parser.add_argument("--attmod", type=str, default="smooth",help="attmod")
    parser.add_argument("--sharp_beta", type=int, default=1,help="sharp_beta")
    
    args = parser.parse_args()
    args.normalizeatt = True
    OUTPATH = args.outputpath
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    
    websiteoutput = open(OUTPATH+"/prediction_predicted_num.txt",'w')
    websiteoutput.write("Start:0\n")
    websiteoutput.close()
    run_model(**vars(args))
    websiteoutput = open(OUTPATH+"/prediction_predicted_num.txt",'w')
    websiteoutput.write("All:100\n")
    websiteoutput.close()



#python3 Multihead_predict.py --dataset "./testdata/Test_fold0.fasta" --outputpath ./Results/5foldresult/
