import random
import functools
import re
#import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from full_imdb_dataloader import IMDB


def conv_len_fn(padding, kernel_size, dilation, stride):
    def conv_len(length):
        return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    return conv_len


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, functions, lambda x: x)


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        only_alphabet = lambda w: re.sub('[^a-z.]', '', w)
        not_empty = lambda w: len(w) != 0
        x = list(filter(not_empty, map(only_alphabet, x)))
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # config
        self.conv_hidden = 128
        self.conv1 = tnn.Conv1d(50, self.conv_hidden, kernel_size=1, padding=5)
        self.conv1_len = conv_len_fn(5, 1, 1, 1)
        self.conv2 = tnn.Conv1d(self.conv_hidden, self.conv_hidden, kernel_size=2, padding=5)
        self.conv2_len = conv_len_fn(5, 2, 1, 1)
        self.mp = tnn.MaxPool1d(2)
        self.mp_len = conv_len_fn(0, 2, 1, 2)

        self.lstm = tnn.LSTM(128, 256, batch_first=True, num_layers=2, dropout=0.5)
        self.fc1 = tnn.Linear(256 * 2, 128)
        self.fc2 = tnn.Linear(128, 1)

        self.do = tnn.Dropout(0.5)
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        C = input.permute((0, 2, 1))
        C = self.mp(self.relu(self.conv1(C)))
        C = self.mp(self.relu(self.conv2(C)))
        C = C.permute((0, 2, 1))
        nl = self.mp_len(self.conv2_len(self.mp_len(self.conv1_len(length))))
        packed = tnn.utils.rnn.pack_padded_sequence(C, nl, batch_first=True)
        _, (L, _) = self.lstm(packed)
        L = self.relu(self.fc1(torch.cat((L[-1, :, :], L[-2, :, :]), dim=1)))
        L = self.do(L)
        L = self.fc2(L)
        return L.squeeze()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="test")
    print(f'{len(train)} in train set')
    print(f'{len(dev)} in dev set')
    # Test everything
    tmp = dev
    dev = train
    train = tmp

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    net.load_state_dict(torch.load('model-split-train-dev.pth', map_location=torch.device(device)))

    num_correct = 0

    print("Loaded model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
