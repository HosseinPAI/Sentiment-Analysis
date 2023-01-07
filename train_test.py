import time
import re
from models import LSTM, pyramid
import torch
import torch.nn as nn
from dataset import word_embedding


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def train_LSTM(data, model, device, criterion, optimizer, batch_size, epoch, word_embedding_data,
               average, padding_size, embedding_dim, mode):
    correct = 0
    data_len = 0
    total_loss = 0
    start_time = time.time()
    print('Training Epoch: {}'.format(epoch))
    hidden = model.init_hidden(batch_size)
    model.train()
    for batch_idx, (batch_sentence) in enumerate(data):
        seq, target = word_embedding(batch_sentence, word_embedding_data, average, padding_size, embedding_dim, device)
        if batch_idx % 10 == 0:
            print('.', end='')
        if len(seq) == batch_size:
            seq = seq.to(device)
            target = target.to(device)
            if mode == 'Pyramid':
                hidden = [tuple([state.data for state in hidden[0]]),
                          tuple([state.data for state in hidden[1]]),
                          tuple([state.data for state in hidden[2]]),
                          tuple([state.data for state in hidden[3]])]
            else:
                hidden = tuple([state.data for state in hidden])
            optimizer.zero_grad()
            target_pred, hidden = model(seq, hidden)
            loss = criterion(target_pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            # -----------------------------------------------------------
            total_loss += loss.item()
            target_prediction = target_pred.argmax(dim=1, keepdim=True)
            correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
            data_len += batch_size

    total_loss /= (batch_idx + 1)
    acc = 100.00 * (correct / data_len)
    print("\nTrain Loss: {:.6f}     Train Accuracy: {:.2f}%        Training Time: {:.2f} min".format(total_loss, acc, (
                time.time() - start_time) / 60.00))
    return total_loss, acc


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def test_LSTM(data, model, device, criterion, batch_size, word_embedding_data,
              average, padding_size, embedding_dim, mode):
    correct = 0
    data_len = 0
    total_loss = 0

    hidden = model.init_hidden(batch_size)
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_sentence) in enumerate(data):
            seq, target = word_embedding(batch_sentence, word_embedding_data, average, padding_size, embedding_dim,
                                         device)
            seq = seq.to(device)
            target = target.to(device)
            if mode == 'Pyramid':
                hidden = [tuple([state.data for state in hidden[0]]),
                          tuple([state.data for state in hidden[1]]),
                          tuple([state.data for state in hidden[2]]),
                          tuple([state.data for state in hidden[3]])]
            else:
                hidden = tuple([state.data for state in hidden])
            target_pred, hidden = model(seq, hidden)
            loss = criterion(target_pred, target)
            # -----------------------------------------------------------
            total_loss += loss.item()
            target_prediction = target_pred.argmax(dim=1, keepdim=True)
            correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
            data_len += batch_size

        total_loss /= (batch_idx + 1)
        acc = 100.00 * correct / data_len
        print("Test Loss : {:.6f}     Test Accuracy : {:.2f}%".format(total_loss, acc))
        print("================================================================================")
        return total_loss, acc, target, target_prediction


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def train_bert(train_data, model, device, criterion, optimizer, epoch):
    correct = 0
    data_len = 0
    total_loss = 0
    start_time = time.time()
    print('Training Epoch: {}'.format(epoch))
    model.train()
    for batch_idx, (data, mask, target) in enumerate(train_data):
        if batch_idx % 250 == 0:
            print('.', end='')

        data = data.to(device)
        mask = mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        target_pred = model(data, token_type_ids=None, attention_mask=mask, labels=target)
        loss = criterion(target_pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        target_prediction = target_pred.argmax(dim=1, keepdim=True)
        correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
        data_len += len(data)

    total_loss /= (batch_idx + 1)
    acc = 100.00 * (correct / data_len)
    print("\nTrain Loss: {:.6f}     Train Accuracy: {:.2f}%        Training Time: {:.2f} min".format(total_loss, acc, (
                time.time() - start_time) / 60.00))
    return total_loss, acc


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def test_bert(test_data, model, device, criterion, batch_size, test_len):
    correct = 0
    data_len = 0
    total_loss = 0
    target_out = torch.empty(test_len)
    target_pred_out = torch.empty(test_len)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, mask, target) in enumerate(test_data):
            data = data.to(device)
            mask = mask.to(device)
            target = target.to(device)
            target_pred = model(data, token_type_ids=None, attention_mask=mask, labels=target)
            loss = criterion(target_pred, target)
            # -----------------------------------------------------------
            total_loss += loss.item()
            target_prediction = target_pred.argmax(dim=1, keepdim=True)
            correct += target_prediction.eq(target.view_as(target_prediction)).sum().item()
            data_len += len(data)

            target_out[batch_idx * batch_size:data_len] = target
            target_pred_out[batch_idx * batch_size:data_len] = target_prediction.resize_(len(data))

        total_loss /= (batch_idx + 1)
        acc = 100.00 * correct / data_len
        print("Test Loss : {:.6f}     Test Accuracy : {:.2f}%".format(total_loss, acc))
        print("================================================================================")
        return total_loss, acc, target_out, target_pred_out


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def prepare_data(input_sentence, word_embedding, padding_size=280):
    output_data = []
    index = 0

    line = input_sentence
    line.strip()
    line = line.lower()
    if line.find('http') == -1 and line.find('://') == -1 and line.find('www') == -1:
        line = re.sub(r'@\w+', 'mention', line)
        line = re.sub(r'#\w+', 'hashtag', line)
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
    return data


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def impose_on_model(sentence, network, padding_size, device, word_embedding, word_embedding_average, mode):
    data_for_model = torch.zeros((1, padding_size, 300))
    for i in range(padding_size):
        index = sentence[i]
        if index == -1:
            data_for_model[0][i] = word_embedding_average
        elif index == 0:
            data_for_model[0][i] = torch.tensor([0 for k in range(300)])
        else:
            data_for_model[0][i] = word_embedding.vectors[index]

    if mode == 0:
        model = LSTM.Sentiment140_LSTM(300, 150, 3, device).to(device)
    elif mode == 1:
        model = LSTM.Sentiment140_Bidirectional_LSTM(300, 150, 3, device).to(device)
    else:
        model = pyramid.Sentiment140_Pyramid_LSTM(300, 150, 3, device).to(device)

    model.load_state_dict(torch.load(network))
    hidden = model.init_hidden(1)
    model.eval()
    target_pred, hidden = model(data_for_model.to(device), hidden)
    target_prediction = target_pred.argmax(dim=1, keepdim=True)

    return target_prediction
