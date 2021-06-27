import csv
import os
import numpy as np


class Gene_data:
    train_test_split_ratio = 0.1
    
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.seq = None
        self.seqleft = None
        self.seqright = None
        self.length = None
        np.random.seed(1234)
    
    @classmethod
    def load_sequence(cls, dataset, left=1000, right=3000,predict=False):
        genes = []
        #count = 0
        path = dataset
        print('Importing dataset {0}'.format(dataset))
        with open(path, 'r') as f:
            index=0
            for line in f:
                if line[0] == '>':
                    if index!=0:
                        seq=seq.upper()
                        seq=seq.replace('U','T')
                        seq=list(seq)
                        #change all other characters into N
                        for index in range(len(seq)):
                            if seq[index] not in ['A','C','G','T']:
                               test=1
                               seq[index]='N'
                        
                        seq = ''.join(seq)
                        
                        seq_length = len(seq)
                        line_left = seq[:int(seq_length*left/(right+left))]
                        line_right = seq[int(seq_length*left/(right+left)):]
                        if len(line_right) >= right:
                            line_right = line_right[-right:]
                        
                        if len(line_left) >= left:
                            line_left = line_left[:left]
                        
                        
                        gene = Gene_data(id,label)
                        gene.seqleft = line_left.rstrip()
                        gene.seqright = line_right.rstrip()
                        gene.length = seq_length
                        #if transcript_biotype != 'protein_coding':
                        #    count += 1
                        genes.append(gene)
                    
                    id = line.strip()
                    label = line[1:].split(',')[0] #changed to label not float
                    seq=""
                else:
                    seq+=line.strip()
                
                #print(index)
                index+=1
            
            #last seq 
            seq=seq.upper()
            seq=seq.replace('U','T')
            seq=list(seq)
            #change all other characters into N
            for index in range(len(seq)):
                if seq[index] not in ['A','C','G','T']:
                   test=1
                   seq[index]='N'
            
            seq = ''.join(seq)
            
            seq_length = len(seq)
            line_left = seq[:int(seq_length*left/(right+left))]
            line_right = seq[int(seq_length*left/(right+left)):]
            if len(line_right) >= right:
                line_right = line_right[-right:]
            
            if len(line_left) >= left:
                line_left = line_left[:left]
            
            gene = Gene_data(id,label)
            gene.seqleft = line_left.rstrip()
            gene.seqright = line_right.rstrip()
            gene.length = seq_length
            genes.append(gene)
        
        genes = np.array(genes)
        if not predict:
           genes = genes[np.random.permutation(np.arange(len(genes)))]
        
        print('Total number of samples:', genes.shape[0])
        return genes

