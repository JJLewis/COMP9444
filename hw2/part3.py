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
from imdb_dataloader import IMDB


def swapsie(sentence, to_swap_idx, max_dist):
    '''
    Single inplace swap
    '''
    distance = random.randint(1, max_dist)
    direction = -1 if random.randint(0, 1) == 0 else 1
    other = to_swap_idx + (distance * direction)
    other = other if other in range(0, len(sentence)) else [0, 0, len(sentence) - 1][direction + 1]
    tmp = sentence[to_swap_idx]
    sentence[to_swap_idx] = sentence[other]
    sentence[other] = tmp


def swapsies_nd(sentence, prob, max_dist):
    new_sentence = sentence.copy()
    did_swap = False
    for i in range(0, len(new_sentence)):
        r = random.uniform(0, 1)
        if r < prob:
            did_swap = True
            swapsie(new_sentence, i, max_dist)
    if not did_swap:
        swapsie(new_sentence, random.randint(0, len(sentence) - 1), max_dist)
    return new_sentence


def swapsies_d(sentence, min_swaps, max_swaps, max_dist):
    new_sentence = sentence.copy()
    num_swaps = round(random.uniform(min_swaps, max_swaps))
    print(num_swaps)
    for _ in range(num_swaps):
        to_swap_idx = random.randint(0, len(new_sentence) - 1)
        swapsie(new_sentence, to_swap_idx, max_dist)
    return new_sentence


def dropsies_d(sentence, min_drops, max_drops, keep_half=False):
    if len(sentence) == 1:
        return sentence
    safe_max = max_drops if len(sentence) / (keep_half + 1) > max_drops else len(sentence) / (keep_half + 1) - (
        not keep_half)
    num_drops = random.randint(min_drops, safe_max)
    new_sentence = sentence.copy()
    for _ in range(num_drops):
        del new_sentence[random.randint(0, len(new_sentence) - 1)]

    return new_sentence


def dropsies_nd(sentence, prob, keep_half=False):
    if len(sentence) == 1:
        return sentence

    new_sentence = []
    for word in sentence:
        r = random.uniform(0, 1)
        if r > prob:
            new_sentence.append(word)

    # if no words left, return a random number of words
    if len(new_sentence) == 0 or (keep_half and len(new_sentence) < len(sentence) / 2):
        new_sentence = sentence.copy()
        for _ in range(random.randint(1, len(new_sentence) / (keep_half + 1) - (not keep_half))):
            del new_sentence[random.randint(0, len(new_sentence) - 1)]

    return new_sentence


def augmentie_nd(sentence, drop_prob, swap_prob, max_dist, keep_half=False):
    new_sentence = swapsies_nd(sentence, swap_prob, max_dist)
    new_sentence = dropsies_nd(new_sentence, drop_prob, keep_half=False)
    return new_sentence


def more(train_set, drop_prob, swap_prob, max_dist, keep_half=False):
    new = []
    for example in train_set:
        ne = data.example.Example()
        ### augment words
        ne.text = augmentie_nd(example.text, drop_prob, swap_prob, max_dist, keep_half=keep_half)
        ne.label = example.label
        new.append(ne)
    return new


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

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    # Augment
    m = more(train, 0.2, 0.2, 5)
    train.examples += m

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    # for plotting accuracy
    acc = []
    epo = []

    for epoch in range(10): # we saved our model at each epoch and chose to submit the one at epoch 6
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

        num_correct = 0
        # net = net.eval()
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
            # Save mode

        # net = net.train()

        accuracy = 100 * num_correct / len(dev)
        acc.append(accuracy)
        epo.append(epoch)

        print(f"Classification accuracy: {accuracy}")
        # torch.save(net.state_dict(), path + str(epoch) + '-' + str(accuracy))
        print("Saved model")

        #plt.plot(epo, acc)
        #plt.show()

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

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
