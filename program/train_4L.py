import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from neural_net_4L import CodemixDataset
from tqdm import tqdm
from neural_net_4L import get_embedding


def get_scores(y_true, y_pred):
    if torch.cuda.is_available():
        accuracy = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
        recall = recall_score(y_true.cpu(), y_pred.cpu(), average='weighted')
        precision = precision_score(y_true.cpu(), y_pred.cpu(), average='weighted')
    else:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
    return accuracy, f1, recall, precision



def train(model_architecture, encoder_model, encoding_type, train_df, test_df, epochs, \
          learning_rate, batch_size, log, device, hidden1,hidden2,hidden3, dropout):
    input_size = get_embedding([train_df.iloc[0]['text']], encoder_model, encoding_type).shape
    model = model_architecture(input_size[1], len(train_df['label'].unique()),hidden1,hidden2,hidden3,dropout).to(device)
    
    train_set = CodemixDataset(train_df, encoder_model, encoding_type)
    val_set = CodemixDataset(test_df, encoder_model, encoding_type)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        output_labels = torch.Tensor().to(device)
        true_labels = torch.Tensor().to(device)
        total_loss = 0
        for data in tqdm(train_loader):
            text, labels = data['text'], data['label']
            inputs = text.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs.type(torch.float))
            predicted_labels = torch.argmax(outputs, dim=1)
            output_labels = torch.cat((output_labels, predicted_labels), 0)
            true_labels = torch.cat((true_labels, labels), 0)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        tr_accuracy, tr_f1, tr_recall, tr_precision = get_scores(true_labels, output_labels)
        net_train_loss = total_loss/(len(train_loader)*batch_size)
        print('Epoch: {}\t Train Loss: {:.4f} \t Train F1:{:.2f}'.format(epoch, net_train_loss, tr_f1), end='\t')
        
        output_labels = torch.Tensor().to(device)
        true_labels = torch.Tensor().to(device)
        total_loss = 0
        for i, data in enumerate(test_loader, 0):
            text, labels = data['text'], data['label']
            
            inputs = text.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs.type(torch.float))
            predicted_labels = torch.argmax(outputs, dim=1)
            output_labels = torch.cat((output_labels, predicted_labels), 0)
            true_labels = torch.cat((true_labels, labels), 0)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
        ts_accuracy, ts_f1, ts_recall, ts_precision = get_scores(true_labels, output_labels)
        net_test_loss = total_loss/(len(train_loader)*batch_size)
        print('Test Loss: {:.4f} \t Test F1:{:.2f}'.format(net_test_loss, ts_f1))

        if log != None:
            log({
                    "train accuracy": tr_accuracy,
                    "train f1": tr_f1,
                    "train recall": tr_recall,
                    "train precision": tr_precision,
                    
                    "test accuarcy": ts_accuracy,
                    "test f1": ts_f1,
                    "test recall": ts_recall,
                    "test precision": ts_precision,
                    
                    "train loss": net_train_loss,
                    "test loss": net_test_loss
                    })