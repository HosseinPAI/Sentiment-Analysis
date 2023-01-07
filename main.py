import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import linecache
import argparse
import os
from dataset import Sentiment140_DataSet, Sentiment140_BERT_dataset
import pandas as pd
import torchtext as tt
from pytorch_pretrained_bert import BertConfig
from models import LSTM, pyramid, bert
from train_test import test_bert, train_bert, test_LSTM, train_LSTM
from plot import show_plot, CM_calculator


if __name__ == '__main__':
    # Create Parameters
    parser = argparse.ArgumentParser(description='Pytorch Fast Human Pose Estimation!')
    parser.add_argument('--extract_path', type=str, default='',
                        help='Glove42B_300d path')
    parser.add_argument('--glove42B_path', type=str, default='original_data/glove.42B.300d.zip',
                        help='Glove42B_300d path')
    parser.add_argument('--original_dataset_path', type=str, default='original_data/trainingandtestdata.zip',
                        help='Original Dataset path')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--result_path', type=str, default='', help='result path')
    parser.add_argument('--mode', type=str, default='LSTM', help='Mode of Training')
    parser.add_argument('--worker', type=int, default=2, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of each batch')
    parser.add_argument('--padding_size', type=int, default=280, help='Padding size')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--test_data_len', type=int, default=367, help='Test data length')
    parser.add_argument('--test_batch_size', type=int, default=367, help='Test batch size')
    parser.add_argument('--train_data_len', type=int, default=150000, help='Train data length')
    parser.add_argument('--num_of_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--epoches', type=int, default=10, help='Number of epoches')

    # Start Process
    args = parser.parse_args()
    current_dir = os.getcwd()
    args.result_path = current_dir + '/results/'
    args.dataset_path = current_dir + '/dataset/'
    args.extract_path = current_dir + '/original_data/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    """
    This Function Has Four Modes, You can Set The Mode variable base on following options
    Mode = LSTM     : Simple LSTM with one hidden layer
    Mode = LSTM_Bid : Bidirectional LSTM
    Mode = Pyramid  : Pyramid LSTM
    Mode = Bert     : Pyramid LSTM
    """
    # =====================================================================================
    # If available use GPU memory to load data
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    kwargs = {'num_workers': args.worker, 'pin_memory': True} if use_cuda else {}

    # Create Embedding Vector
    word_embedding_data = tt.vocab.GloVe(name='42B', dim=300, cache=args.extract_path)
    average = sum(word_embedding_data.vectors) / len(word_embedding_data.vectors)

    if args.mode == 'Bert':
        train_set = pd.read_csv(args.dataset_path+'bert_train_dataset.csv', index_col=None,
                                header=None, names=['Polarity', 'Tweet'])
        test_set = pd.read_csv(args.dataset_path+'bert_test_dataset.csv', index_col=None,
                               header=None, names=['Polarity', 'Tweet'])
        args.batch_size = 16
        args.padding_size = 300
        args.test_data_len = 367
        train_data = train_set['Tweet']
        train_label = train_set['Polarity']
        test_data = test_set['Tweet']
        test_label = test_set['Polarity']
        train_lists = [train_data.values.tolist(), train_label.values.tolist()]
        test_lists = [test_data.values.tolist(), test_label.values.tolist()]

        train_dataset = Sentiment140_BERT_dataset(train_lists, args.padding_size)
        test_dataset = Sentiment140_BERT_dataset(test_lists, args.padding_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        args.batch_size = 16#500
        args.padding_size = 280
        args.embedding_dim = 300
        args.test_data_len = 367
        args.test_batch_size = args.test_data_len
        args.train_data_len = 367# 150000
        linecache.clearcache()
        test_dataset = Sentiment140_DataSet(args.dataset_path+'test_dataset.csv', args.test_data_len,
                                            args.padding_size)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        linecache.clearcache()
        train_dataset = Sentiment140_DataSet(args.dataset_path+'train_dataset.csv', args.train_data_len,
                                             args.padding_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Create Model and Other Parameters
    if args.mode == 'Bert':
        # Model
        config = BertConfig(vocab_size_or_config_json_file=32000)
        model = bert.BertForSequenceClassification(config, args.num_of_classes).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam([
            {"params": model.bert.parameters(), "lr": 0.00001},
            {"params": model.classifier.parameters(), "lr": 0.001}])
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    else:
        if args.mode == 'LSTM':
            model = LSTM.Sentiment140_LSTM(device, input_dim=300, hidden_dim=150,
                                           output_dim=args.num_of_classes).to(device)
            learning_rate = 0.003
            w_decay = 0.00001
            print('Training Simple LSTM Network for Sentimetn140 Dataset is Started.')
        elif args.mode == 'LSTM_Bid':
            model = LSTM.Sentiment140_Bidirectional_LSTM(device, input_dim=300, hidden_dim=150,
                                                         output_dim=args.num_of_classes).to(device)
            print('Training Bidirectional LSTM Network for Sentimetn140 Dataset is Started.')
            learning_rate = 0.001
            w_decay = 0.00001
        else:
            model = pyramid.Sentiment140_Pyramid_LSTM(device, input_dim=300, hidden_dim=64,
                                                      output_dim=args.num_of_classes).to(device)
            print('Training Pyramid LSTM Network for Sentimetn140 Dataset is Started.')
            learning_rate = 0.001
            w_decay = 0.00001

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    print("================================================================================")
    epoches = 7
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    true_class = []
    pred_class = []

    if args.mode == 'Bert':
        args.epoches = 4
        for i in range(args.epoches):
            train_l, train_a = train_bert(train_loader, model, device, criterion, optimizer, i + 1)
            test_l, test_a, true_class, pred_class = test_bert(test_loader, model, device, criterion,
                                                               args.batch_size, args.test_data_len)
            scheduler.step()
            train_loss.append(train_l)
            train_acc.append(train_a)
            test_loss.append(test_l)
            test_acc.append(test_a)
    else:
        args.epoches = 10
        for i in range(epoches):
            train_l, train_a = train_LSTM(train_loader, model, device, criterion, optimizer,
                                          args.batch_size, i + 1, word_embedding_data, average,
                                          args.padding_size, args.embedding_dim, args.mode)

            test_l, test_a, true_class, pred_class = test_LSTM(test_loader, model, device, criterion,
                                                               args.test_batch_size, word_embedding_data,
                                                               average, args.padding_size,
                                                               args.embedding_dim, args.mode)
            scheduler.step()
            train_loss.append(train_l)
            train_acc.append(train_a)
            test_loss.append(test_l)
            test_acc.append(test_a)

    # Save Trained Model
    torch.save(model.state_dict(), args.result_path + "{0}_Model.pt".format(args.mode))
    # Show The Results
    CM_calculator(true_class, pred_class, args.test_batch_size, args.result_path, args.mode)
    show_plot(train_loss, test_loss, train_acc, test_acc, args.result_path, args.mode)
