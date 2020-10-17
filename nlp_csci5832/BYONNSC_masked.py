import random
from random import shuffle
random.seed(11)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(11)
from transformers import RobertaTokenizerFast, RobertaModel
    
    
class MyNeuralClassifier(nn.Module):
    """
    This is a neural classifier.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        """
        Note: if your model needs other parameters, feel free to change them.
        """
        super(MyNeuralClassifier, self).__init__()

        # TODO: implement your preferred architecture!

        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.embedding = RobertaModel.from_pretrained('roberta-base').embeddings

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            batch_first=True,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=0.3,
            bidirectional=True
        )
        self.predictor = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)
        
    def forward(self, input):
        # TODO: implement your preferred architecture!

        tokenized_dict = self.tokenizer.encode_plus(
            text=input,
            add_special_tokens=True,
            pad_to_max_length=True,
            max_length=75,
            return_attention_mask=True,
            truncation=True
        )

        input_ids = torch.tensor(tokenized_dict["input_ids"])
        embedding = self.embedding.word_embeddings(input=input_ids)
        embedding = embedding.view(-1, embedding.shape[0], embedding.shape[1])

        hidden, state = self.lstm.forward(input=embedding)

        # out = torch.stack([context[0][0], context[0][1]], dim=1)
        out = torch.cat([state[0][0], state[0][1]], dim=1)

        logits = self.predictor(input=out)
        logits = logits.float()
        # probs = nn.functional.softmax(logits, dim=1)
        # y_hat = torch.argmax(probs, dim=1)

        # if labels is not None:
        #     result["loss"] = nn.functional.cross_entropy(logits, labels)
        # result["logits"] = logits
        # result["probs"] = probs
        # result["py_index"] = py_index
        # return result
        return logits
        
    
def load_data(fname='BYOSC_data.csv'):
    """
    This function loads the data. The default value for fname assumes that
    the file with the data is in the same folder as this python file.
    If this isn't true, change the value of fname to the location of your data.
    """
    data = []
    for line in open(fname, encoding="utf8", errors="ignore"):
        # This only loads the columns that we care about.
        y, _, _, _, _, x = line[1:-2].strip().split('","')
        data.append((x, y))
    
    # This shuffles the dataset. shuffle() gives you the data in a random order.
    # However, we have set random.seed(11) above, so this should give the same
    # order every time you run this code.
    shuffle(data)
    
    # We will use the first 300 examples for training and the rest as our dev set.
    # (We will not use a test set; you can assume that would be held-out data.)
    train_set = data[:300]
    dev_set = data[300:]

    return train_set, dev_set
    

def make_feature_vectors(data, w2i):
    """
    This function takes text data and returns vectors with the indices of the words.
    """
    new_data = []

    for (x, y) in data:
        sentence = []
        for word in x.split(' '):
            if word in w2i:
                sentence.append(w2i[word])
            else:
                sentence.append(w2i["<UNK>"])

        new_data.append((torch.tensor(sentence), torch.tensor([int(int(y)/2)])))
    
    return new_data


def get_labels(data):
    """
    This function takes text data and returns vectors with the indices of the words.
    """
    labels = []

    for (x, y) in data:

        labels.append(torch.tensor([int(int(y) / 2)]))

    return labels


def eval(model, data):
    """
    This function evaluates the model on the given data.
    It prints the accuracy.
    """
    # Set the model to evaluation mode; no updates will be performed.
    # (The opposite command is model.train().)
    model.eval()
    
    total = right = 0
    for (x, y) in data:
        # model(x) calls the forward() function of the model.
        # Here is the point where we manually add a softmax function.
        probs = F.softmax(model(x), dim=1)
        y_hat = torch.argmax(probs, dim=1)
        if y_hat == int(int(y)/2):
            right += 1
        total += 1
        
    print("Accuracy: " + str((right * 1.0)/total))


def print_params(model):
    """
    This function can be used to print (and manually inspect) the model parameters.
    """
    for name, param in model.named_parameters():
        print(name)
        print(param)
        print(param.grad)


def get_vocab(data):
    w2i = {'<UNK>': 0}

    for (x, y) in data:
        for word in x.split(' '):
            if word not in w2i:
                w2i[word] = len(w2i)

    return w2i
    

if __name__ == "__main__":
    """
    This is the entry point for the code.
    """
    # This loads the training and development set.
    train, dev = load_data()
    # This constructs the vocabulary.
    vocab = get_vocab(train)
    # train_labels = get_labels(train)
    # dev_labels = get_labels(dev)
    # train = make_feature_vectors(train, vocab)
    # dev = make_feature_vectors(dev, vocab)
    
    # This creates the classifier (with random parameters!).
    # TODO: substitute -1 by more reasonable values!
    model = MyNeuralClassifier(vocab_size=len(vocab.keys()), emb_dim=768, hidden_dim=200, output_dim=3)
    
    # The next 2 lines define the loss function and the optimizer.
    # SGD = stochastic gradient descent
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    # This is a sanity check: what is the model performance on train and dev?
    # The model parameters are currently random, so the performance should be, too.
    eval(model, train)
    eval(model, dev)
    print()
        
    # The next part is the actual model training.
    epochs = 5  # how many times does the model see the entire training set?
    for i in range(epochs):
        print("Starting epoch " + str(i))
        # The next line ensures that model parameters are being updated.
        # (The opposite command is model.eval().)
        model.train()
        for (x, y) in train:
            model.zero_grad()
            # Compute the model's prediction for the current example.
            raw_scores = model(x)
            # Compute the loss from the prediction and the gold label.
            # loss = loss_function(raw_scores.unsqueeze(0), torch.tensor(int(y)))
            loss = loss_function(raw_scores, torch.tensor([int(int(y)/2)]))
            print(loss)
            # Compute the gradients.
            loss.backward()
            # Update the model; the parameters should change during this step.
            optimizer.step()
        
        # This is a second sanity check.
        # The model performance should increase after every epoch!
        eval(model, train)
        eval(model, dev)
        print()
        
    
