import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from train import train
from neural_net import LinearNetwork_2Layer, LinearNetwork_4Layer
import wandb
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from helper import get_arguments, get_data
from pymagnitude import Magnitude
from neural_net import CodemixDataset
import os

arguments = get_arguments()

tags = [
        ('tamilmixsentiment', None, 'ta'), 
        ('offenseval_dravidian', 'tamil', 'ta'), 
        ('offenseval_dravidian', 'malayalam', 'ml'),
        ('offenseval_dravidian', 'kannada', 'kn'),
        ('kan_hope', None, 'kn')
        ]

tag_dict = {
    'tamilmixsentiment': tags[0],
    'offenseval_dravidian_ta': tags[1],
    'offenseval_dravidian_ml': tags[2],
    'offenseval_dravidian_kn': tags[3],
    'kan_hope': tags[4]
}

dataset_name = arguments['dataset']
tag = tag_dict[dataset_name]

EPOCHS = 50
ETA = 0.001
BATCH_SIZE = 64
tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp_dropout=[0,0.25]
mlp_hiddenNodes_combinations = [[1024, 2048], [(1024,512,256),(512,256,128)]]
ngram_analyzer = ["char","word"]
ngram_ranges = [(1,3),(2,5)]
ngram_maxfeatures = [1000, 2000, 5000, 10000]



for na in ngram_analyzer:
    for nr in ngram_ranges:
        for nmf in ngram_maxfeatures:
            for hiddenNode_idx, num_layers in enumerate([2,3]):
                mlp_hiddenNodes = mlp_hiddenNodes_combinations[hiddenNode_idx]
                for dp in mlp_dropout: #Dropout
                    for hn in mlp_hiddenNodes: #hidden nodes
                        
                        train_df, val_df = get_data(tag)


                        # fname = arguments['model_name']+"_"+tag[0]+"_"+tag[2]+"_"+str(len(mlp_hiddenNodes))+"L_"+str(dp)+"D_"+str(hn)
                        fname = arguments['model_name']+tag[0]+"_"+tag[2]+"_"+na+"_"+str(nr[0])+"-"+str(nr[1])+"_"+str(nmf)+"_"+str(num_layers)+"L_"+str(dp)+"D_"+str(hn)
                        print(fname)

                        if num_layers == 2:
                            LinearNetwork = LinearNetwork_2Layer
                        elif num_layers == 3:
                            LinearNetwork = LinearNetwork_4Layer

                        #change project name    #entity - nnproj
                        run = wandb.init(project=dataset_name, entity="nnproj",reinit=True,name=fname)
                        
                        #change enbedding model
                        if arguments['model_name'] == 'TfidfVectorizer':
                            vectorizer = TfidfVectorizer(analyzer=na, ngram_range=nr, max_features=nmf)
                            vectorizer.fit(train_df['text'])
                        elif arguments['model_name'] == 'CountVectorizer':
                            vectorizer = CountVectorizer(analyzer=na, ngram_range=nr, max_features=nmf)
                            vectorizer.fit(train_df['text'])

                        train_set = CodemixDataset(train_df, vectorizer, arguments['type'])
                        val_set = CodemixDataset(val_df, vectorizer, arguments['type'])
                        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
                        test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
                        train(LinearNetwork, vectorizer, arguments['type'], train_df, val_df, epochs=EPOCHS, \
                            learning_rate=ETA, batch_size=BATCH_SIZE, log=wandb.log, device=tdevice,hiddenNodes=hn,dropout=dp, \
                            train_loader=train_loader, test_loader=test_loader)            
                        
                        run.finish()
