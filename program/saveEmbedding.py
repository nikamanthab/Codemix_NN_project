import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from train import train
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from helper import get_arguments, get_data
from pymagnitude import Magnitude
from neural_net import CodemixDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from neural_net import CodemixDataset
from tqdm import tqdm
from neural_net import get_embedding

arguments = get_arguments()

# arguments={
#         'weights_path': 'oasndonasd',
#         'model_name': 'TfidfVectorizer',
#         'type': 'TfidfVectorizer',
#         'dataset': 'tamilmixsentiment',
#         'analyzer': 'char',
#         'range': (1,3),
#         'max_features': 1000
#     }

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

# for tag in tags: #dataset

train_df, val_df = get_data(tag)
train_size = train_df.shape[0]
val_size = val_df.shape[0]

BATCH_SIZE=1
#change embedding model
encoder_model = None
if arguments['type'] == 'magnitude':
    english_path = os.path.join(arguments['weights_path'], 'english', arguments['model_name']+'.magnitude')
    dravidian_path = os.path.join(arguments['weights_path'], 'dravidian', tag[2], arguments['model_name']+'.magnitude')

    english_model = Magnitude(english_path)
    dravidian_model = Magnitude(dravidian_path)
    encoder_model = Magnitude(english_model, dravidian_model, devices=[0,1])
elif arguments['type'] == 'BERT':
    encoder_model = SentenceTransformer('distiluse-base-multilingual-cased')
elif arguments['type'] == 'TfidfVectorizer':
    encoder_model = TfidfVectorizer(analyzer=arguments['analyzer'], ngram_range=arguments['range'], max_features=arguments['max_features'])
    encoder_model.fit(train_df['text'])
elif arguments['type'] == 'CountVectorizer':
    encoder_model = CountVectorizer(analyzer=arguments['analyzer'], ngram_range=arguments['range'], max_features=arguments['max_features'])
    encoder_model.fit(train_df['text'])

train_set = CodemixDataset(train_df, encoder_model, arguments['type'])
val_set = CodemixDataset(val_df, encoder_model, arguments['type'])


train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

encoding = get_embedding([train_df.iloc[0]['text']], encoder_model, arguments['type'])[0]

train_embedding_size = (train_size,1,encoding.shape[0])
train_embedding=torch.zeros(train_embedding_size)
for i, data in enumerate(tqdm(train_loader)):
    text, labels = data['text'], data['label']
    train_embedding[i] = text

train_filename = '../embeddingoutputs/trainEmbedding_'+dataset_name+'_'+arguments['type']+'.pt'
print(train_filename)
torch.save(train_embedding, train_filename)

val_embedding_size = (val_size,1,encoding.shape[0])
val_embedding=torch.zeros(val_embedding_size)
for i, data in enumerate(test_loader, 0):
    text, labels = data['text'], data['label']
    val_embedding[i] = text

val_filename = '../embeddingoutputs/valEmbedding_'+dataset_name+'_'+arguments['type']+'.pt'
print(val_filename)
torch.save(val_embedding, val_filename)


    
