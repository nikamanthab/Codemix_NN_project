from datasets import list_datasets, load_dataset
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from train import train
from neural_net import LinearNetwork
import wandb
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tags = [#('tamilmixsentiment', None, 'tamil'), 
        #('offenseval_dravidian', 'tamil', 'tamil'), 
        #('offenseval_dravidian', 'malayalam', 'malayalam'),
        ('offenseval_dravidian', 'kannada', 'kannada'),
        ('kan_hope', None, 'kannada')
        #('Shushant/NepaliSentiment', None, 'nepali')
        ]

def get_data(tag):
    dataset = load_dataset(tag[0], tag[1])
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df['text'] = dataset['train']['text']
    train_df['label'] = dataset['train']['label']
    
    val_df['text'] = dataset['validation']['text']
    val_df['label'] = dataset['validation']['label']
    
    # print("TRAIN DESCRIPTION:")
    # print("Value Counts:")
    # print(train_df['label'].value_counts())
          
    # print("Sample text and label:")
    # for i in range(5):
    #     idx = random.randint(0, len(train_df))
    #     sample = train_df.iloc[idx]
    #     print("Text: {}, Label:{}".format(sample['text'], sample['label']))
    return train_df, val_df

EPOCHS = 10
ETA = 0.001
BATCH_SIZE = 32
tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_df, val_df = get_data(tags[1])
# train_df = pd.read_csv('../dataset/dravidian-codemix/tamil_train.tsv', sep='\t')
# val_df = pd.read_csv('../dataset/dravidian-codemix/tamil_dev.tsv', sep='\t')


mlp_dropout=[0,0.25]
mlp_hiddenNodes = [1024, 2048]

for tag in tags:
    train_df, val_df = get_data(tag)
    for dp in mlp_dropout:
        for hn in mlp_hiddenNodes:
            fname = tag[0]+"_"+tag[2]+"_2L_"+str(dp)+"D_"+str(hn)
            print(fname)
            run = wandb.init(project="BERT_MLP2", entity="nnproj",reinit=True,name=fname)
            
            model = SentenceTransformer('distiluse-base-multilingual-cased')
            train(LinearNetwork, model, 'BERT', train_df, val_df, epochs=EPOCHS, \
                  learning_rate=ETA, batch_size=BATCH_SIZE, log=wandb.log, device=tdevice,hiddenNodes=hn,dropout=dp)
            

run.finish()
                    
# CountVectorizer vec
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,3), max_features=2048)
# vectorizer.fit(train_df['text'])
# train(LinearNetwork, vectorizer, 'CountVectorizer', train_df, val_df, epochs=EPOCHS, \
#       learning_rate=ETA, batch_size=BATCH_SIZE, log=wandb.log, device=tdevice)

# # tfidf vec
# vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=2048)
# vectorizer.fit(train_df['text'])
# train(LinearNetwork, vectorizer, 'TfidfVectorizer', train_df, val_df, epochs=EPOCHS, \
#       learning_rate=ETA, batch_size=BATCH_SIZE, log=None, device='cpu')

# #fasttext - specific language
# model = Magnitude("../weights/fasttext/{}/{}.magnitude".format(tags[0][2], tags[0][2]))
# train(LinearNetwork, model, 'magnitude', train_df, val_df, epochs=EPOCHS, \
#       learning_rate=ETA, batch_size=BATCH_SIZE, log=None, device='cpu')

# bert - multilingual
# model = SentenceTransformer('distiluse-base-multilingual-cased')
# train(LinearNetwork, model, 'BERT', train_df, val_df, epochs=EPOCHS, \
#       learning_rate=ETA, batch_size=BATCH_SIZE, log=None, device='cpu')

