from keras import regularizers
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Embedding, BatchNormalization, Input, \
    concatenate, Multiply, Dot, Reshape, Activation, Lambda, Masking,concatenate,Add

from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import MaxPooling1D as layers_maxpooling1d
from keras.backend import int_shape


from hier_attention_mask import Attention_mask
from keras.models import Model
from six.moves import range
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef
import subprocess
from keras import backend as K
from keras.initializers import random_normal
import os
import scipy.stats as stats
import csv
import sys

OUTPATH = None
def margin_loss(y_true,y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.1 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    
    return K.mean(K.sum(L, 1))

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  
  Args:
    input_tensor: float Tensor to perform activation.
  
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) *K.binary_crossentropy(y_true, y_pred),
            axis=-1)
    
    return weighted_loss

def conv1d_bn(x,
              filters,
              filters_length,
              dropout,
              padding='same',
              strides=1,
              use_bias=False,
              BN=False,
              active=True,
              active_function='relu',
              pre_activation=False,
              ):
    if strides is None:
        strides = 1
    else:
        if type(strides) is int:
            strides = strides
    if pre_activation:
       if BN:
          x = BatchNormalization()(x)#BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
       if active:
          x = Activation(active_function)(x)
    
    
    x = Convolution1D(
        filters,
        filters_length,
        init='he_normal',
        strides=strides,
        padding=padding,#padding=padding, for new version of keras
        use_bias=use_bias#for new version of keras
        )(x)
    
    if not pre_activation:
        if BN:
            x = BatchNormalization()(x)#BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    
    x = Dropout(dropout)(x)
    
    if not pre_activation:
        if active:
           x = Activation(active_function)(x)
    
    return x

def residual_layer_2016(indata, out_channel, i_bn=False,k1 = 3,k2 = 3, strides = None,dropout=0,actfun='relu'):
    conv_out1 = conv1d_bn(indata, out_channel,k1, dropout=dropout, padding='same',use_bias=False,BN=i_bn,active_function=actfun,pre_activation=True)
    conv_out2 = conv1d_bn(conv_out1, out_channel,k2, dropout=dropout, padding='same',use_bias=False,BN=i_bn,active_function=actfun,pre_activation=True)
    relu_out = Add()([indata,conv_out2])
    return relu_out


def reshape_pyramidal(outputs):  #, sequence_length):
                """
                Reshapes the given outputs, i.e. reduces the
                time resolution by 2.
                Similar to "Listen Attend Spell".
                https://arxiv.org/pdf/1508.01211.pdf
                """
                # [batch_size, max_time, num_units]
                shape = outputs.get_shape().as_list()
                #shape = tf.shape(outputs)
                max_time = shape[1]
                num_units = outputs.get_shape().as_list()[-1]
                pads = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
                outputs = tf.pad(outputs, pads)
                concat_outputs = Reshape((int(np.ceil(max_time/2)),num_units * 2 ))(outputs)
                return concat_outputs  #, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)

class multihead_attention:
    
    def __init__(self, max_len, nb_classes, save_path, kfold_index):
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path
        self.kfold_index = kfold_index
    
    def build_model_multihead_attention_multiscaleCNN4_covermore(self,
                                        dim_attention,headnum,
                                        embedding_vec,
                                        load_weights=False, weight_dir=None,
                                        nb_filters=32,filters_length1=1,
                                        filters_length2=5,
                                        filters_length3=10,
                                        pooling_size=3,
                                        drop_input=0,
                                        drop_cnn=0.2,
                                        drop_flat=0,
                                        W1_regularizer=0.005,
                                        W2_regularizer=0.005,
                                        Att_regularizer_weight=0.0005,
                                        BatchNorm=False,
                                        fc_dim = 50,
                                        fcnum=0,
                                        posembed=False,
                                        pos_dmodel=40,
                                        pos_nwaves = 20,
                                        posmod = 'concat',
                                        regularfun=1,
                                        huber_delta=1,
                                        activation='gelu',
                                        activationlast='gelu',
                                        add_avgpooling = False,
                                        poolingmod=1,
                                        normalizeatt=False,
                                        regressionmodel=False,
                                        attmod = "softmax",
                                        sharp_beta=1,
                                        lr = 0.001 
                                        ):
        """
        same as build_model_multihead_attention_multiscaleCNN4_dropout except change l2 reguliza to l1
        """
        ###print('Advanced Masking')
        def mask_func(x):
            return x[0] * x[1]
        
        ###print(posembed)
        ###print(posmod)
        input = Input(shape=(self.max_len,), dtype='int8')
        input_mask = Input(shape=([int(self.max_len/pooling_size), 1]), dtype='float32')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = Dropout(drop_input)(embedding_layer(input)) #layer2
        if 'gelu' in activation:
           activationfun=gelu
        else:
           activationfun = 'relu'
        
        if 'gelu' in activationlast:
            activationlastfun = gelu
        else:
            activationlastfun='relu'
        
        ###print(activationfun)
        ###print(activationlastfun)
        with tf.name_scope('first_cnn'):
            first_cnn = Convolution1D(nb_filters, filters_length1, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationfun, use_bias=False,name='CNN1')(embedding_output) #layer3
            first_cnn2 = Convolution1D(int(nb_filters/2), filters_length1, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationlastfun, use_bias=False)(first_cnn) #layer5
            second_cnn = Convolution1D(nb_filters, filters_length2, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationfun, use_bias=False,name='CNN2')(embedding_output) #layer4
            second_cnn2 = Convolution1D(int(nb_filters/2), filters_length2, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationlastfun, use_bias=False)(second_cnn)
            third_cnn = Convolution1D(int(nb_filters/2), filters_length3, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationfun, use_bias=False,name='CNN3')(embedding_output)
            
            third_cnn2 = Convolution1D(int(nb_filters/2), filters_length3, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='same', activation=activationlastfun, use_bias=False)(third_cnn)
            if BatchNorm:
                 first_cnn2 = BatchNormalization()(first_cnn2)
                 second_cnn2 = BatchNormalization()(second_cnn2)
                 third_cnn2 = BatchNormalization()(third_cnn2)
            
            if not add_avgpooling:
                if poolingmod == 1:
                    pooling_layer = MaxPooling1D(pool_length=pooling_size, stride=pooling_size)
                else:
                    pooling_layer = AveragePooling1D(pool_length=pooling_size, stride=pooling_size)
                
                cnn_output1 = Dropout(drop_cnn)(pooling_layer(first_cnn2))
                cnn_output2 = Dropout(drop_cnn)(pooling_layer(second_cnn2))
                cnn_output3 = Dropout(drop_cnn)(pooling_layer(third_cnn2))
            else:
               first_cnn2_max=MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(first_cnn2)
               first_cnn2_avg=AveragePooling1D(pool_length=pooling_size, stride=pooling_size)(first_cnn2)
               cnn_output1 = Dropout(drop_cnn)(concatenate([first_cnn2_max,first_cnn2_avg],axis=-1))
               second_cnn2_max=MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(second_cnn2)
               second_cnn2_avg=AveragePooling1D(pool_length=pooling_size, stride=pooling_size)(second_cnn2)
               cnn_output2 = Dropout(drop_cnn)(concatenate([second_cnn2_max,second_cnn2_avg],axis=-1))
               third_cnn2_max=MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(third_cnn2)
               third_cnn2_avg=AveragePooling1D(pool_length=pooling_size, stride=pooling_size)(third_cnn2)
               cnn_output3 = Dropout(drop_cnn)(concatenate([third_cnn2_max,third_cnn2_avg],axis=-1))
               
               
        
        if posembed:
            ##print(posmod)
            from position_embedding import PositionEmbedding
            if posmod == 'concat':
                pos_emb1 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=pos_nwaves, d_model=pos_dmodel,name='pos_emb1')(cnn_output1)
                cnn_output1 = concatenate([cnn_output1, pos_emb1], axis=-1)
                pos_emb2 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=pos_nwaves, d_model=pos_dmodel,name='pos_emb2')(cnn_output2)
                cnn_output2 = concatenate([cnn_output2, pos_emb2], axis=-1)
                pos_emb3 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=pos_nwaves, d_model=pos_dmodel,name='pos_emb3')(cnn_output3)
                cnn_output3 = concatenate([cnn_output3, pos_emb3], axis=-1)
            else:
                ##print("yes add posmod")
                pos_emb1 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=int(int_shape(cnn_output1)[-1]/2), d_model=int_shape(cnn_output1)[-1],name='pos_emb1')(cnn_output1)
                cnn_output1 = Add()([cnn_output1, pos_emb1])
                pos_emb2 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=int(int_shape(cnn_output2)[-1]/2), d_model=int_shape(cnn_output2)[-1],name='pos_emb2')(cnn_output2)
                cnn_output2 = Add()([cnn_output2, pos_emb2])
                pos_emb3 = PositionEmbedding(max_time=int(self.max_len/pooling_size), n_waves=int(int_shape(cnn_output3)[-1]/2), d_model=int_shape(cnn_output3)[-1],name='pos_emb3')(cnn_output3)
                cnn_output3 = Add()([cnn_output3, pos_emb3])
        
        mask_input1 = []
        mask_input1.append(cnn_output1)
        mask_input1.append(input_mask)
        cnn_mask_output1 = Lambda(mask_func)(mask_input1)
        del mask_input1
        mask_input2 = []
        mask_input2.append(cnn_output2)
        mask_input2.append(input_mask)
        cnn_mask_output2 = Lambda(mask_func)(mask_input2)
        del mask_input2
        mask_input3 = []
        mask_input3.append(cnn_output3)
        mask_input3.append(input_mask)
        cnn_mask_output3 = Lambda(mask_func)(mask_input3)
        del mask_input3
        
        if regularfun==1:
           regularizerfunction_W1 = regularizers.l1(W1_regularizer)
           regularizerfunction_W2 = regularizers.l1(W2_regularizer)
        elif regularfun==2:
           regularizerfunction_W1 = regularizers.l2(W1_regularizer)
           regularizerfunction_W2 = regularizers.l2(W2_regularizer)
        elif regularfun ==3:
           regularizerfunction_W1 = smoothL1(W1_regularizer,huber_delta)
           regularizerfunction_W2 = smoothL1(W2_regularizer,huber_delta)
        
        with tf.name_scope('multihead_attention'):
            att1,att1_A = Attention_mask(hidden=cnn_output1.get_shape()[-1].value, da=dim_attention, r=headnum, init='glorot_uniform', activation='tanh',
                    W1_regularizer=regularizerfunction_W1,
                    W2_regularizer=regularizerfunction_W2,
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,sharp_beta=sharp_beta,name="att1")(concatenate([cnn_mask_output1, input_mask]))#-5 layer
            
            att2,att2_A = Attention_mask(hidden=cnn_output1.get_shape()[-1].value, da=dim_attention, r=headnum, init='glorot_uniform', activation='tanh',
                    W1_regularizer=regularizerfunction_W1,
                    W2_regularizer=regularizerfunction_W2,
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,sharp_beta=sharp_beta,name="att2")(concatenate([cnn_mask_output2, input_mask])) #-4 layer
            
            att3,att3_A = Attention_mask(hidden=cnn_output1.get_shape()[-1].value, da=dim_attention, r=headnum, init='glorot_uniform', activation='tanh',
                    W1_regularizer=regularizerfunction_W1,
                    W2_regularizer=regularizerfunction_W2,
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,sharp_beta=sharp_beta,name="att3")(concatenate([cnn_mask_output3, input_mask])) #-3 layer
            
            if BatchNorm:
               att1 = BatchNormalization()(att1)
               att2 = BatchNormalization()(att2)
               att3 = BatchNormalization()(att3)
               
            
            output = Dropout(drop_flat)(Flatten()(concatenate([att1,att2,att3]))) #-2 layer
        
        fc = output
        for _ in range(fcnum):
             fc = Dense(fc_dim,activation='relu')(fc)
             fc = Dropout(drop_flat)(fc)
        
        with tf.name_scope(''):
            if regressionmodel:
              preds = Dense(self.nb_classes,activation='softmax')(fc) #-1 layer
            else:
              preds = Dense(self.nb_classes,activation='sigmoid')(fc) #-1 layer
        
        self.model = Model(inputs=[input,input_mask], outputs=preds)
        from keras import optimizers
        # optim = optimizers.RMSprop()
        optim = optimizers.Adam(lr=lr, decay=5e-5) #The paper uses a decay rate alpha = alpha/sqrt(t) updted each epoch (t) for the logistic regression demonstration.
        #optim = optimizers.nadam()
        #optim = RAdam()
        if regressionmodel:
            self.model.compile(loss='kld',optimizer=optim,metrics=['acc'])
        else:
            self.model.compile(loss='binary_crossentropy',optimizer=optim,metrics=['binary_accuracy','categorical_accuracy'])
        
        
        
        if load_weights:
            self.model.load_weights(weight_dir)
        
        self.is_built = True
        self.bn = False
        self.model.summary()
    
    @classmethod
    def acc(cls, y_true, y_pred):
        '''
        soft-accuracy; should never be used.
        :param y_true: target probability mass of mRNA samples
        :param y_pred: predcited probability mass of mRNA samples
        :return: averaged accuracy
        '''
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
    
    def get_feature(self, X):
        '''
        K.learning_phase() returns a binary flag
        The learning phase flag is a bool tensor (0 = test, 1 = train)
        to be passed as input to any Keras function that
        uses a different behavior at train time and test time.
        '''
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _convout1_f = K.function(inputs, [self.model.layers[2].output])  # output of first convolutional filter
        activations = _convout1_f([0] + [X])
        
        return activations
    
    def get_attention(self, X):
        """
        Get the output of attention module, which assigns weights to different parts of sequence.
        from the Activation('softmax') layer
        :param X: samples for weights attention weights will be extracted
        :return:
        """
        if self.bn:
            layer = 16
        else:
            layer = 14
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _attention_f = K.function(inputs, [
            self.model.layers[layer].output])
        
        return _attention_f([0] + [X])
    
    def get_masking(self, X):
        if self.bn:
            layer = 14
        else:
            layer = 12
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _attention_f = K.function(inputs, [self.model.layers[layer].output])
        
        return _attention_f([0] + [X])
    
    def get_attention_multiscale_batch(self, X,X_mask):
        """
        Get the output of attention module, which assigns weights to different parts of sequence.
        from the Activation('softmax') layer
        :param X: samples for weights attention weights will be extracted
        :return:
        """
        layer = -3
        attmodel1 = Model(self.model.inputs,self.model.get_layer('att1').output[1])
        attmodel2 = Model(self.model.inputs,self.model.get_layer('att2').output[1])
        attmodel3 = Model(self.model.inputs,self.model.get_layer('att3').output[1])
        return attmodel1.predict([X,X_mask.reshape(-1,X_mask.shape[1],1)],batch_size=100),attmodel2.predict([X,X_mask.reshape(-1,X_mask.shape[1],1)],batch_size=100),attmodel3.predict([X,X_mask.reshape(-1,X_mask.shape[1],1)],batch_size=100)
    
    def train(self, x_train, y_train, mask_label, batch_size, epochs=100,x_valid=None,y_valid=None,mask_valid=None,loadFinal=False,classweight=False,class_weights=None):
        if not self.is_built:
            print('Run build_model() before calling train opertaion.')
            return
        
        ##print("begin to train\n")
        if x_valid is None:
            size_train = len(x_train)
            x_valid = x_train[int(0.9 * size_train):]
            y_valid = y_train[int(0.9 * size_train):]
            x_train = x_train[:int(0.9 * size_train)]
            y_train = y_train[:int(0.9 * size_train)]
            mask_valid = mask_label[int(0.9 * size_train):]
            mask_train = mask_label[:int(0.9 * size_train)]
        else:
            mask_train = mask_label
        
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(self.kfold_index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, verbose=1)
        ###print(self.model.evaluate(x_train, y_train, batch_size=batch_size))
        ###print(self.model.evaluate(x_valid, y_valid, batch_size=batch_size))
        ##print("before training")
        if classweight:
               #if self.nb_classes == 6:
               #    class_weights={0:1,1:1,2:1,3:3,4:5,5:8}
               #if self.nb_classes ==5:  #13490,
               #    class_weights={0:1,1:1,2:2,3:4,4:6}
               
               hist = self.model.fit([x_train,mask_train.reshape(-1,mask_train.shape[1],1)], y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_data=([x_valid,mask_valid.reshape(-1,mask_valid.shape[1],1)], y_valid), callbacks=[model_checkpoint,early_stopping], class_weight=class_weights,shuffle=True)
        
        else:
              hist = self.model.fit([x_train,mask_train.reshape(-1,mask_train.shape[1],1)], y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_data=([x_valid,mask_valid.reshape(-1,mask_valid.shape[1],1)], y_valid), callbacks=[model_checkpoint,early_stopping], shuffle=True)
        
        ##print("after training")
        # load best performing model
        if not loadFinal:
            self.model.load_weights(best_model_path)
        
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('categorical_accuracy'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_categorical_accuracy'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(self.kfold_index), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(self.kfold_index), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(self.kfold_index), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(self.kfold_index), Valid_Acc, delimiter=',')
    
    def train_optimizor(self, x_train, y_train, mask_train,x_test,y_test,mask_test,batch_size, epochs=100):
        if not self.is_built:
            ##print('Run build_model() before calling train opertaion.')
            return
        size_train = len(x_train)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        hist = self.model.fit([x_train, mask_train.reshape(-1,x_train.shape[1],1)],y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_data=([x_test,mask_test.reshape(-1,x_test.shape[1],1)], y_test), shuffle=True)
        
        Valid_Loss = np.asarray(hist.history['val_loss'])
        Valid_Acc = np.asarray(hist.history['val_acc'])
        
        return Valid_Loss,Valid_Acc
    
    
    def evaluate(self,x_test,y_test,mask_label):
        import pickle
        pred_y = self.model.predict([x_test,mask_label.reshape(-1,mask_label.shape[1],1)])
        y_label_ = list()
        nclass = pred_y.shape[1]
        roc_auc = dict()
        average_precision = dict()
        #binary_acc=[]
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        mcc_dict=dict()
        for i in range(nclass):#calculate one by one
            average_precision[i+1] = average_precision_score(y_test[:, i], pred_y[:, i])
            roc_auc[i+1] = roc_auc_score(y_test[:,i], pred_y[:,i])
            mcc_dict[i+1] = matthews_corrcoef(y_test[:,i],[1 if x>0.5 else 0 for x in pred_y[:,i]])
            fpr[i],tpr[i],_ = roc_curve(y_test[:,i], pred_y[:,i])
            precision[i],recall[i],_ =  precision_recall_curve(y_test[:, i], pred_y[:, i])
            
            #binary_acc.append(accuracy_score(y_test[:,i],[if for x in pred_y[:,i]]))
        
        fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(), pred_y.ravel())
        precision['micro'],recall['micro'],_ =  precision_recall_curve(y_test.ravel(), pred_y.ravel())
        
        average_precision["micro"] = average_precision_score(y_test, pred_y,average="micro")
        roc_auc["micro"] = roc_auc_score(y_test,pred_y,average="micro")
        roc_list = [roc_auc[x+1] for x in range(nclass)]
        roc_list.append(roc_auc['micro'])
        pr_list = [average_precision[x+1] for x in range(nclass)]
        pr_list.append(average_precision['micro'])
        mcc_list = [mcc_dict[x+1] for x in range(nclass)]
        np.savetxt(OUTPATH + 'testevaluation_roc_average_presicion_fold_{}.txt'.format(self.kfold_index), np.array(roc_list+pr_list+mcc_list), delimiter=',')
        picklefile = open(OUTPATH + '5foldavg_test_for_plot','wb')
        pickle.dump((fpr,tpr,precision,recall),picklefile)
        return roc_auc,average_precision
    
    def evaluate_regression(self,x_test,y_test,mask_label):
        from scipy.stats import pearsonr
        import pickle
        label_code = [[0,0,0,1],
                      [0,0,1,0],
                      [0,1,0,0],
                      [1,0,0,0]
                     ]
        
        pred_y = self.model.predict([x_test,mask_label.reshape(-1,mask_label.shape[1],1)])
        y_label_ = list()
        nclass = pred_y.shape[1]
        roc_auc = dict()
        average_precision = dict()
        pearsoncrr=dict()
        #binary_acc=[]
        y_test_binary = []
        for label in y_test:
            y_test_binary.append(label_code[np.argmax(label)])
        
        y_test_binary = np.asarray(y_test_binary)
        fpr = dict()
        tpr = dict()
        mcc_dict=dict()
        precision = dict()
        recall = dict()
        for i in range(nclass):#calculate one by one
            average_precision[i+1] = average_precision_score(y_test_binary[:, i], pred_y[:, i])
            roc_auc[i+1] = roc_auc_score(y_test_binary[:,i], pred_y[:,i])
            mcc_dict[i+1] = matthews_corrcoef(y_test_binary[:,i],[1 if x>0.5 else 0 for x in pred_y[:,i]])
            fpr[i],tpr[i],_ = roc_curve(y_test_binary[:,i], pred_y[:,i])
            precision[i],recall[i],_ =  precision_recall_curve(y_test_binary[:, i], pred_y[:, i])
            pearsoncrr[i+1] = pearsonr(y_test[:,i],pred_y[:,i])
            #binary_acc.append(accuracy_score(y_test_binary[:,i],[if for x in pred_y[:,i]]))
        
        fpr['micro'],tpr['micro'],_ = roc_curve(y_test_binary.ravel(), pred_y.ravel())
        precision['micro'],recall['micro'],_ =  precision_recall_curve(y_test_binary.ravel(), pred_y.ravel())
        
        average_precision["micro"] = average_precision_score(y_test_binary, pred_y,average="micro")
        roc_auc["micro"] = roc_auc_score(y_test_binary,pred_y,average="micro")
        roc_list = [roc_auc[x+1] for x in range(nclass)]
        roc_list.append(roc_auc['micro'])
        pr_list = [average_precision[x+1] for x in range(nclass)]
        pr_list.append(average_precision['micro'])
        mcc_list = [mcc_dict[x+1] for x in range(nclass)]
        np.savetxt(OUTPATH + 'testevaluation_roc_average_presicion_fold_{}.txt'.format(self.kfold_index), np.array(roc_list+pr_list+mcc_list), delimiter=',')
        picklefile = open(OUTPATH + '5foldavg_test_for_plot','wb')
        pickle.dump((fpr,tpr,precision,recall),picklefile)
        
        output = open(OUTPATH + 'testevaluation_pearson_fold_{}.txt'.format(self.kfold_index),'w')
        for i in range(nclass):
            ##print("pearsoncrr for nclass:"+str(i)+" is "+str(pearsoncrr[i+1][0])+"\n")
            output.write(str(i)+":"+str(pearsoncrr[i+1][0])+":"+str(pearsoncrr[i+1][1])+"\n")
        
        output.close()
        
        return roc_auc,average_precision
    
    def train_regression(self, x_train, y_train, mask_label, batch_size, epochs=100,x_valid=None,y_valid=None,mask_valid=None,loadFinal=False):
        if not self.is_built:
            ##print('Run build_model() before calling train opertaion.')
            return
        
        ##print("begin to train\n")
        if x_valid is None:
            size_train = len(x_train)
            x_valid = x_train[int(0.9 * size_train):]
            y_valid = y_train[int(0.9 * size_train):]
            x_train = x_train[:int(0.9 * size_train)]
            y_train = y_train[:int(0.9 * size_train)]
            mask_valid = mask_label[int(0.9 * size_train):]
            mask_train = mask_label[:int(0.9 * size_train)]
        else:
            mask_train = mask_label
        
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(self.kfold_index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, verbose=1)
        ###print(self.model.evaluate(x_train, y_train, batch_size=batch_size))
        ###print(self.model.evaluate(x_valid, y_valid, batch_size=batch_size))
        hist = self.model.fit([x_train,mask_train.reshape(-1,mask_train.shape[1],1)], y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_data=([x_valid,mask_valid.reshape(-1,mask_valid.shape[1],1)], y_valid), callbacks=[model_checkpoint,early_stopping], shuffle=True)
        
        # load best performing model
        if not loadFinal:
            self.model.load_weights(best_model_path)
        
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(self.kfold_index), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(self.kfold_index), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(self.kfold_index), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(self.kfold_index), Valid_Acc, delimiter=',')
    
    def get_encodings(self,X):
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _encoding_f = K.function(inputs, [self.model.layers[1].output])
        return _encoding_f([0] + [X])
    
    def get_PCM_multiscale_weighted(self,X,mask_label,nb_filters,filters_length1,filters_length2,filters_length3):
        ###for membery efficient
        onehotX = self.get_encodings(X)[0]
        feature_model1 = Model(self.model.inputs,self.model.get_layer('CNN1').output)#layer 2 is cnn for 1CNN model no dropout!
        feature_model2 = Model(self.model.inputs,self.model.get_layer('CNN2').output)
        feature_model3 = Model(self.model.inputs,self.model.get_layer('CNN3').output)
        
        def add(feature_length,up=True):
             if up:
                return int((feature_length-1)/2)
             
             else:
                 return feature_length-1-int((feature_length-1)/2)
        
        Add1up = add(filters_length1,True)
        Add1down = add(filters_length1,False)
        Add2up = add(filters_length2,True)
        Add2down = add(filters_length2,False)
        Add3up = add(filters_length3,True)
        Add3down = add(filters_length3,False)
        
        #CNNoutputs1=feature_model1.predict(X,batch_size=50)#S,8000,32 
        #CNNoutputs2=feature_model2.predict(X,batch_size=50)#S,8000,32
        #CNNoutputs3=feature_model3.predict(X,batch_size=50)#S,8000,16
        for m in range(nb_filters):
            PCM1=np.zeros((filters_length1,4))
            PCM2=np.zeros((filters_length2,4))
            PCM3=np.zeros((filters_length3,4))
            for s in range(len(X)):
                #if s%1000==0:
                #     print(s)
                
                CNNoutputs1=feature_model1.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                CNNoutputs2=feature_model2.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                sub_index1=CNNoutputs1[0,:,m].argmax()-Add1up
                sub_index2=CNNoutputs2[0,:,m].argmax()-Add2up
                if m<int(nb_filters/2):
                   CNNoutputs3=feature_model3.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                   sub_index3=CNNoutputs3[0,:,m].argmax()-Add3up
                
                if CNNoutputs1[0,:,m].max()>0:
                    if sub_index1>=0 and sub_index1+filters_length1<onehotX.shape[1]:
                       PCM1 = PCM1+onehotX[s,sub_index1:(sub_index1+filters_length1),:]*CNNoutputs1[0,:,m].max()
                    elif sub_index1<0:
                       PCM1 = PCM1+np.pad(onehotX[s,0:sub_index1+filters_length1,:],([-sub_index1,0],[0,0]),'constant',constant_values =0)*CNNoutputs1[0,:,m].max() #add zeros before
                       
                    else:
                       PCM1 = PCM1+np.pad(onehotX[s,sub_index1:,:],([0,filters_length1-onehotX.shape[1]+sub_index1],[0,0]),'constant',constant_values =0)*CNNoutputs1[0,:,m].max() #add zeros after
                if CNNoutputs2[0,:,m].max()>0:
                    if sub_index2>=0 and sub_index2+filters_length2<onehotX.shape[1]:
                       PCM2 = PCM2+onehotX[s,sub_index2:(sub_index2+filters_length2),:]*CNNoutputs2[0,:,m].max()
                    elif sub_index2<0:
                       PCM2 = PCM2+np.pad(onehotX[s,0:sub_index2+filters_length2,:],([-sub_index2,0],[0,0]),'constant',constant_values =0)*CNNoutputs2[0,:,m].max() #add zeros before
                    else:
                       PCM2 = PCM2+np.pad(onehotX[s,sub_index2:,:],([0,filters_length2-onehotX.shape[1]+sub_index2],[0,0]),'constant',constant_values =0)*CNNoutputs2[0,:,m].max() #add zeros after
                if m < int(nb_filters/2):
                   if CNNoutputs3[0,:,m].max()>0:
                    if sub_index3>=0 and sub_index3+filters_length3<onehotX.shape[1]:
                       PCM3 = PCM3+onehotX[s,sub_index3:(sub_index3+filters_length3),:]*CNNoutputs3[0,:,m].max()
                    elif sub_index3<0:
                       PCM3 = PCM3+np.pad(onehotX[s,0:sub_index3+filters_length3,:],([-sub_index3,0],[0,0]),'constant',constant_values =0)*CNNoutputs3[0,:,m].max() #add zeros before
                    else:
                       PCM3 = PCM3+np.pad(onehotX[s,sub_index3:,:],([0,filters_length3-onehotX.shape[1]+sub_index3],[0,0]),'constant',constant_values =0)*CNNoutputs3[0,:,m].max() #add zeros after
                
            np.savetxt(OUTPATH + '/PCMmultiscale_weighted_filter1_{}.txt'.format(m), PCM1, delimiter=',')
            np.savetxt(OUTPATH + '/PCMmultiscale_weighted_filter2_{}.txt'.format(m), PCM2, delimiter=',')
            if m < int(nb_filters/2):
               np.savetxt(OUTPATH + '/PCMmultiscale_weighted_filter3_{}.txt'.format(m), PCM3, delimiter=',')
    
    def get_PCM_multiscale(self,X,mask_label,nb_filters,filters_length1,filters_length2,filters_length3):
        ###for membery efficient
        onehotX = self.get_encodings(X)[0]
        feature_model1 = Model(self.model.inputs,self.model.get_layer('CNN1').output)#layer 2 is cnn for 1CNN model no dropout!
        feature_model2 = Model(self.model.inputs,self.model.get_layer('CNN2').output)
        feature_model3 = Model(self.model.inputs,self.model.get_layer('CNN3').output)
        
        def add(feature_length,up=True):
             if up:
                return int((feature_length-1)/2)
             
             else:
                 return feature_length-1-int((feature_length-1)/2)
        
        Add1up = add(filters_length1,True)
        Add1down = add(filters_length1,False)
        Add2up = add(filters_length2,True)
        Add2down = add(filters_length2,False)
        Add3up = add(filters_length3,True)
        Add3down = add(filters_length3,False)
        
        #CNNoutputs1=feature_model1.predict(X,batch_size=50)#S,8000,32 
        #CNNoutputs2=feature_model2.predict(X,batch_size=50)#S,8000,32
        #CNNoutputs3=feature_model3.predict(X,batch_size=50)#S,8000,16
        for m in range(nb_filters):
            PCM1=np.zeros((filters_length1,4))
            PCM2=np.zeros((filters_length2,4))
            PCM3=np.zeros((filters_length3,4))
            for s in range(len(X)):
                #if s%1000==0:
                #     print(s)
                
                CNNoutputs1=feature_model1.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                CNNoutputs2=feature_model2.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                
                sub_index1=CNNoutputs1[0,:,m].argmax()-Add1up
                sub_index2=CNNoutputs2[0,:,m].argmax()-Add2up
                if m<int(nb_filters/2):
                    CNNoutputs3=feature_model3.predict([X[s:s+1],mask_label[s:s+1].reshape(-1,mask_label[s:s+1].shape[1],1)],batch_size=50)
                    sub_index3=CNNoutputs3[0,:,m].argmax()-Add3up
                
                if CNNoutputs1[0,:,m].max()>0:
                    if sub_index1>=0 and sub_index1+filters_length1<onehotX.shape[1]:
                       PCM1 = PCM1+onehotX[s,sub_index1:(sub_index1+filters_length1),:]
                    elif sub_index1<0:
                       PCM1 = PCM1+np.pad(onehotX[s,0:sub_index1+filters_length1,:],([-sub_index1,0],[0,0]),'constant',constant_values =0) #add zeros before
                       
                    else:
                       PCM1 = PCM1+np.pad(onehotX[s,sub_index1:,:],([0,filters_length1-onehotX.shape[1]+sub_index1],[0,0]),'constant',constant_values =0) #add zeros after
                if CNNoutputs2[0,:,m].max()>0:
                    if sub_index2>=0 and sub_index2+filters_length2<onehotX.shape[1]:
                       PCM2 = PCM2+onehotX[s,sub_index2:(sub_index2+filters_length2),:]
                    elif sub_index2<0:
                       PCM2 = PCM2+np.pad(onehotX[s,0:sub_index2+filters_length2,:],([-sub_index2,0],[0,0]),'constant',constant_values =0) #add zeros before
                    else:
                       PCM2 = PCM2+np.pad(onehotX[s,sub_index2:,:],([0,filters_length2-onehotX.shape[1]+sub_index2],[0,0]),'constant',constant_values =0) #add zeros after
                if m < int(nb_filters/2):
                   if CNNoutputs3[0,:,m].max()>0:
                    if sub_index3>=0 and sub_index3+filters_length3<onehotX.shape[1]:
                       PCM3 = PCM3+onehotX[s,sub_index3:(sub_index3+filters_length3),:]
                    elif sub_index3<0:
                       PCM3 = PCM3+np.pad(onehotX[s,0:sub_index3+filters_length3,:],([-sub_index3,0],[0,0]),'constant',constant_values =0) #add zeros before
                    else:
                       PCM3 = PCM3+np.pad(onehotX[s,sub_index3:,:],([0,filters_length3-onehotX.shape[1]+sub_index3],[0,0]),'constant',constant_values =0) #add zeros after
                
            np.savetxt(OUTPATH + '/PCMmultiscale_filter1_{}.txt'.format(m), PCM1, delimiter=',')
            np.savetxt(OUTPATH + '/PCMmultiscale_filter2_{}.txt'.format(m), PCM2, delimiter=',')
            if m < int(nb_filters/2):
               np.savetxt(OUTPATH + '/PCMmultiscale_filter3_{}.txt'.format(m), PCM3, delimiter=',')
    
    

