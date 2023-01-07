import numpy as np
import pandas as pd
import torchtext as tt
import re
import torch
import time
from zipfile import ZipFile
from torch.utils.data import Dataset
import linecache
import csv
from pytorch_pretrained_bert import BertTokenizer


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
def unzip_files(glove_dir, dataset_dir, extract_path):
    with ZipFile(glove_dir, 'r') as zip1:
        zip1.extractall(path=extract_path)

    with ZipFile(dataset_dir, 'r') as zip2:
        zip2.extractall(path=extract_path)


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
def prepare_data(data_set, word_embedding, padding_size=280):
    output_data = []
    index = 0

    for i in range(len(data_set)):
        if i % 10000 == 0:
            print('{} Data is Processed.'.format(i))

        polarity = data_set['Polarity'][i]
        line = data_set['Tweet'][i]
        line.strip()
        line = line.lower()
        if line.find('http') == -1 and line.find('://') == -1 and line.find('www') == -1:
            if polarity == 0:
                text = 'negative'
                new_polarity = 0
            elif polarity == 2:
                text = 'natural'
                new_polarity = 1
            else:
                text = 'positive'
                new_polarity = 2

            line = re.sub(r'@\w+', text, line)
            line = re.sub(r'#\w+', text, line)
            line = re.sub(r'[^a-zA-Z0-9]', r' ', line)
            words = line.split()
            temp = torch.empty(len(words))
            index = 0
            for j in range(len(words)):
                if words[j] in word_embedding.itos:
                    temp[index] = word_embedding.itos.index(words[j])
                else:
                    temp[index] = -1
                index += 1

            # correct Padding
            data = torch.empty(padding_size)
            data[-len(words):] = temp
            for j in range(padding_size - len(words)):
                data[j] = 0

            data = data.int()
            output_data.append((data, new_polarity))

    return output_data


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
def data_save(data_set, save_path, padding_size=280):
    save_data = np.zeros((len(data_set), padding_size + 1))
    for i in range(len(data_set)):
        for j in range(padding_size):
            save_data[i][j] = int(data_set[i][0][j])
        save_data[i][j + 1] = int(data_set[i][1])

    np.savetxt(save_path, save_data, delimiter=',', fmt='%d')


# -------------------------------------------------------------------------------
#             Clean Data (Remove tweet with Link, Correct # and @)
# -------------------------------------------------------------------------------
def clean_data_for_bert(data_set):
    output_data_df = pd.DataFrame(columns=['Polarity', 'Tweet'])
    index = 0

    for i in range(len(data_set)):
        if i % 25000 == 0:
            print('{} Data is Processed.'.format(i))

        polarity = data_set['Polarity'][i]
        line = data_set['Tweet'][i]
        line.strip()
        line = line.lower()
        if line.find('http') == -1 and line.find('://') == -1 and line.find('www') == -1:
            if polarity == 0:
                text = 'negative'
                new_polarity = 0
            elif polarity == 2:
                text = 'natural'
                new_polarity = 1
            else:
                text = 'positive'
                new_polarity = 2

            line = re.sub(r'@\w+', text, line)
            line = re.sub(r'#\w+', text, line)
            output_data_df.loc[index] = [new_polarity, line]
            index += 1

    return output_data_df


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
def prep_clean_data(glove_path, original_dataset_path, extract_path, dataset_path, mode='LSTM'):
    start_time = time.time()
    # Unzip Data
    unzip_files(glove_path, original_dataset_path, extract_path)

    # Read Glove42B_300d
    word_embedding_data = tt.vocab.GloVe(name='42B', dim=300, cache=extract_path)

    # Read CSV Files
    if mode == 'Bert':
        input_test_data = pd.read_csv(extract_path+'testdata.manual.2009.06.14.csv', index_col=None, header=None,
                                      usecols=[0, 5], names=['Polarity', 'Tweet'], engine='python',
                                      encoding="ISO-8859-1")
        input_train_data = pd.read_csv(extract_path+'training.1600000.processed.noemoticon.csv', index_col=None,
                                       header=None, usecols=[0, 5], names=['Polarity', 'Tweet'], engine='python',
                                       encoding="ISO-8859-1")
        print("Data cleaning for Bert mode is started ...")
        clean_test_set = clean_data_for_bert(input_test_data)
        clean_test_set.to_csv(dataset_path + '/bert_test_dataset.csv', header=False, index=False)
        print('Test data is prepared and saved as CSV file.')
        print('-----------------------------------------\n')
        clean_train_set = clean_data_for_bert(input_train_data)
        clean_train_set.to_csv(dataset_path+'/bert_train_dataset.csv', header=False, index=False)
        print('Train data is prepared and saved as CSV file.')
        print("Data cleaning for Bert mode is finished. Total Time is : "
              "{:.2f} min".format((time.time() - start_time) / 60.00))
    else:
        input_test_data = pd.read_csv(extract_path+'testdata.manual.2009.06.14.csv', index_col=None, header=None,
                                      names=['Polarity', 'ID', 'Date', 'Query', 'User', 'Tweet'],
                                      encoding="ISO-8859-1")
        input_train_data = pd.read_csv(extract_path+'training.1600000.processed.noemoticon.csv', index_col=None,
                                       header=None, names=['Polarity', 'ID', 'Date', 'Query', 'User', 'Tweet'],
                                       engine='python', encoding="ISO-8859-1")
        # Prepare Test Data
        print('Preparing Test Data is Started ...')
        test_data = prepare_data(input_test_data, word_embedding_data, padding_size=280)
        data_save(test_data, dataset_path+'/test_dataset.csv', padding_size=280)
        print('Test data is prepared and saved as CSV file.')
        print('-----------------------------------------\n')

        # prepare Train Data
        print('Preparing Train Data is Started ...')
        train_data = prepare_data(input_train_data, word_embedding_data, padding_size=280)
        data_save(train_data, dataset_path+'/train_dataset.csv', padding_size=280)
        print('Train data is prepared and saved as CSV file.')

        print('Total Time for the First Step of Preparing Data = {:.2f}'.format((time.time() - start_time) / 60.00))


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
class Sentiment140_DataSet(Dataset):
    def __init__(self, path_file, data_len, padding_size):
        self.path_file = path_file
        self.data_len = data_len
        self.padding_size = padding_size

    def __getitem__(self, idx):
        line = linecache.getline(self.path_file, idx + 1)
        csv_line = csv.reader([line])
        return next(csv_line)

    def __len__(self):
        return self.data_len


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
def word_embedding(batch_sentence, word_embeding, average_word_embeding, padding_size, dim, device):
    # Claculate Batch Size
    b_size = len(batch_sentence[0])
    data = torch.zeros((b_size, padding_size, dim)).to(device)
    target = torch.empty(b_size).to(device)

    for i in range(b_size):
        for j in range(padding_size):
            index = int(batch_sentence[j][i])
            if index == -1:
                data[i][j] = average_word_embeding
            elif index == 0:
                data[i][j] = torch.tensor([0 for k in range(dim)])
            else:
                data[i][j] = word_embeding.vectors[index]

        target[i] = int(batch_sentence[padding_size][i])

    target = target.int()
    target = target.long()
    return (data, target)


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Sentiment140_BERT_dataset(Dataset):
    def __init__(self, data, padding_size):
        self.data = data
        self.padding_size = padding_size

    def __getitem__(self, index):
        tokenized_tweet = tokenizer.tokenize(self.data[0][index])
        if len(tokenized_tweet) > self.padding_size:
            tokenized_tweet = tokenized_tweet[:self.padding_size]

        tweet_ids = tokenizer.convert_tokens_to_ids(tokenized_tweet)
        tweet_ids = [101] + tweet_ids + [102]
        attention_masks = [1] * len(tweet_ids)
        padding = [0] * (self.padding_size - len(tweet_ids))
        tweet_ids += padding
        attention_masks += padding
        tweet_ids = torch.tensor(tweet_ids)
        attention_masks = torch.tensor(attention_masks)
        polarity = torch.tensor((self.data[1][index]))

        return tweet_ids, attention_masks, polarity

    def __len__(self):
        return len(self.data[0])