import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from nltk import word_tokenize


def get_embedding(text, model, model_type):
    if model_type in ['TfidfVectorizer', 'CountVectorizer']:
        return model.transform(text).toarray()
    elif model_type == 'BERT':
        print("BERT model embedding...")
        return model.encode(text, batch_size=8, show_progress_bar=True)
    elif model_type == 'magnitude':
        vectors = []
        for sentence in tqdm(text):
            vectors.append(np.average(model.query(word_tokenize(sentence)), axis=0))
        return vectors  

    
class CodemixDataset(Dataset):
    def __init__(self, df, encoder_model, encoding_type):
        self.df = df
        print("Creating Embedding... Will take some time...")
        self.embedding = get_embedding(list(self.df['text']), \
                                      encoder_model, encoding_type)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text_embedding = self.embedding[idx]
        label = self.df.iloc[idx]['label']
        sample = {
            'text': text_embedding,
            'label': label
        }
        return sample


class LinearNetwork_2Layer(nn.Module):
    def __init__(self, input_size, output_size,hn,p):
        super().__init__()
        # define layers
        self.fc1 = nn.Linear(in_features=input_size,out_features=hn)
        self.fc2 = nn.Linear(in_features=hn,out_features=output_size)
        self.dropout = nn.Dropout(p)
    # define forward function
    def forward(self, t):
        # fc 1
        t=self.fc1(t)
        t=F.relu(t)
        t = self.dropout(t)
        # fc 2
        t=self.fc2(t)
        # don't need softmax here since we'll use cross-entropy as activation.
        return t

class LinearNetwork_4Layer(nn.Module):
    def __init__(self, input_size, output_size,hn,p):
        hn1, hn2, hn3 = hn
        super().__init__()
        # define layers
        self.fc1 = nn.Linear(in_features=input_size,out_features=hn1)
        self.fc2 = nn.Linear(in_features=hn1,out_features=hn2)
        self.fc3 = nn.Linear(in_features=hn2,out_features=hn3)
        self.fc4 = nn.Linear(in_features=hn3,out_features=output_size)
        self.dropout = nn.Dropout(p)
    # define forward function
    def forward(self, t):
        # fc 1
        t=self.fc1(t)
        t=F.relu(t)
        t = self.dropout(t)
        # fc 2
        t=self.fc2(t)
        t=F.relu(t)
        t = self.dropout(t)
         # fc 3
        t=self.fc3(t)
        t=F.relu(t)
        t = self.dropout(t)
         # fc 4
        t=self.fc4(t)
        # don't need softmax here since we'll use cross-entropy as activation.
        return t