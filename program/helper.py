from datasets import list_datasets, load_dataset
import argparse
import pandas as pd

def get_arguments():
    parser = argparse.ArgumentParser(description='Runner module.')
    parser.add_argument('--type', type=str, default='magnitude')
    parser.add_argument('--model_name', type=str, default='glove')
    parser.add_argument('--weights_path', type=str, default='/users/PAS2056/appiahbalaji2/courses/nn/codemix/weights/')
    parser.add_argument('--dataset', type=str, default='kan_hope')

    args = parser.parse_args()

    cmdargs = {
        "type": args.type,
        "model_name": args.model_name,
        "weights_path": args.weights_path,
        'dataset': args.dataset
    }
    return cmdargs




def get_data(tag):
    dataset = load_dataset(tag[0], tag[1])
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df['text'] = dataset['train']['text']
    train_df['label'] = dataset['train']['label']
    
    if tag[0] == 'kan_hope':
        val_df['text'] = dataset['test']['text']
        val_df['label'] = dataset['test']['label']
    else:
        val_df['text'] = dataset['validation']['text']
        val_df['label'] = dataset['validation']['label']

    if tag[0] == 'offenseval_dravidian' and tag[2] == 'ml':
        train_df['label'].replace(5, 4, inplace=True)
        val_df['label'].replace(5, 4, inplace=True)
    
    return train_df[:], val_df[:] #change it back to use full dataset

