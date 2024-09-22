# Team Name - Simplexity
# -*- coding: utf-8 -*-
"""NLPProject.ipynb"""

#Team Members
'''Sarthak Parakh - 115073201
Kushagra Agarwal - 115699197'''

#General Description
'''We implement a baseline model for Named Entity Recognition (NER) using a Bidirectional Long Short-Term Memory (BiLSTM) model with a Conditional Random Field (CRF) layer. NER is a task in Natural Language Processing (NLP) where the goal is to classify named entities in text into predefined categories such as person names, organizations, locations, etc.

1. Dataset Exploration: We utilize the CoNLL 2012 OntoNotes version english v12 dataset, which contains labeled sentences with named entities. We explore the dataset to understand its structure and characteristics, including the number of unique named entity labels, the distribution of sentence lengths, and the distribution of named entities along with POS tags.

2. Data Preprocessing: Before training the model, we preprocess the data by encoding words and named entity labels into numerical indices. We create dictionaries to map words and labels to their respective indices and perform padding to ensure uniform input sizes for the model.

3. Model Architecture: The model architecture consists of an embedding layer, a bidirectional LSTM layer, a linear layer, and a CRF layer. The embedding layer converts input word indices into dense vector representations. The bidirectional LSTM layer captures contextual information from both directions of the input sequence. The linear layer maps LSTM outputs to the number of output classes (named entity labels). Finally, the CRF layer models the sequential dependencies between labels.

4. Training: We train the model using the training data, optimizing it to minimize the cross-entropy loss. The training loop iterates over the dataset for a specified number of epochs, computing gradients and updating model parameters using the Adam optimizer.

5. Evaluation: After training, we evaluate the model's performance on the test dataset. We calculate precision, recall, and F1-score metrics for each class (named entity label) and compute the non-weighted average across all classes to assess overall performance.
'''

#Class Concepts
'''
I. Syntax|Classification: Classification is utilized in the model architecture, where the task of assigning named entity labels to words in the input text is essentially a classification problem. The model learns to classify each word into one of several predefined named entity classes.

II. Semantics|Probabilistic Model: Semantics is implicitly involved in the model's training process. The BiLSTM-CRF model, combined with the CRF layer, can be viewed as a probabilistic model. The CRF layer models the transition probabilities between different named entity labels, capturing the semantics of how labels are sequenced in natural language text.

III. Language Modeling|Transformers: Language modeling is not explicitly utilized in the provided code. However, the concept of language modeling, particularly the idea of capturing contextual information from surrounding words, is inherent in the BiLSTM architecture used in the model. The bidirectional nature of the LSTM allows the model to consider both past and future words when making predictions, similar to the context-awareness of transformer models.

IV. Applications|Custom Statistical or Symbolic: The application domain is Named Entity Recognition (NER), a fundamental task in natural language processing. The code implements a custom statistical model, combining LSTM and CRF layers, to solve this problem. Additionally, data preprocessing, model training, evaluation, and result analysis are all part of the application of NER using the developed model.'''

#Running Environment
'''The below code was run on Google Cobal with T4 GPU'''

#!pip install datasets
#!pip install torchcrf
#!pip install pytorch-crf

import os, sys, random, re, collections, string
import numpy as np
import math
import csv
import sklearn.model_selection
import sklearn.metrics
import heapq
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from datasets import load_dataset
dataset = load_dataset("conll2012_ontonotesv5",'english_v12')

#dataset
'''
DatasetDict({
    train: Dataset({
        features: ['document_id', 'sentences'],
        num_rows: 10539
    })
    validation: Dataset({
        features: ['document_id', 'sentences'],
        num_rows: 1370
    })
    test: Dataset({
        features: ['document_id', 'sentences'],
        num_rows: 1200
    })
})
'''

print("Named Entities in the dataset:")
label_names = dataset['train'].features['sentences'][0]['named_entities'].feature.names
print(label_names)
print("Number of Entities:",len(label_names))

'''
Named Entities in the dataset:
['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE']
Number of Entities: 37
'''

print(dataset['train'].features['sentences'][0]['pos_tags'].feature.names)

'''
['XX', '``', '$', "''", '*', ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB', 'WDT', 'WP', 'WP$', 'WRB']
'''

print("Number of Training Sentences",len(dataset['train']))

print("Sample sentence: ",dataset['train'][3]['sentences'][0]['words'])
print("Sample sentence entities with entities ID: ",dataset['train'][3]['sentences'][0]['named_entities'])

sentences=[]
ner=[]
for i in dataset['train']:
  for j in range(len(i['sentences'])):
    #print(i['sentences'][j]['named_entities'])
    sentences.append(i['sentences'][j]['words'])
    ner.append(i['sentences'][j]['named_entities'])

sentences_val=[]
ner_val=[]
for i in dataset['validation']:
  for j in range(len(i['sentences'])):
    #print(i['sentences'][j]['named_entities'])
    sentences_val.append(i['sentences'][j]['words'])
    ner_val.append(i['sentences'][j]['named_entities'])

sentences_test=[]
ner_test=[]
for i in dataset['test']:
  for j in range(len(i['sentences'])):
    #print(i['sentences'][j]['named_entities'])
    sentences_test.append(i['sentences'][j]['words'])
    ner_test.append(i['sentences'][j]['named_entities'])

print(sentences[5],ner[5])
print(sentences_val[5],ner_val[5])
print(sentences_test[6],ner_test[6])

# Flatten the lists
flat_words = [w for sublist in sentences for w in sublist]
flat_ner = [tag for sublist in ner for tag in sublist]
sentence_ids = [i+1 for i, sublist in enumerate(sentences) for _ in sublist]

# Create DataFrame
df = pd.DataFrame({'SentenceID': sentence_ids, 'Word': flat_words, 'NER': flat_ner})

flat_words_val = [w for sublist in sentences_val for w in sublist]
flat_ner_val = [tag for sublist in ner_val for tag in sublist]
sentence_ids_val = [i+1 for i, sublist in enumerate(sentences_val) for _ in sublist]

# Create DataFrame
df_val = pd.DataFrame({'SentenceID': sentence_ids_val, 'Word': flat_words_val, 'NER': flat_ner_val})

flat_words_test = [w for sublist in sentences_test for w in sublist]
flat_ner_test = [tag for sublist in ner_test for tag in sublist]
sentence_ids_test = [i+1 for i, sublist in enumerate(sentences_test) for _ in sublist]

# Create DataFrame
df_test = pd.DataFrame({'SentenceID': sentence_ids_test, 'Word': flat_words_test, 'NER': flat_ner_test})

print(df)
print(df_val)
print(df_test)

print("Total number of sentences in the training dataset: {:,}".format(df["SentenceID"].nunique()))
print("Total words in the training dataset: {:,}".format(df.shape[0]))

print("Total number of sentences in the validation dataset: {:,}".format(df_val["SentenceID"].nunique()))
print("Total words in the validation dataset: {:,}".format(df_val.shape[0]))

print("Total number of sentences in the test dataset: {:,}".format(df_test["SentenceID"].nunique()))
print("Total words in the test dataset: {:,}".format(df_test.shape[0]))

#trainingdata
ner_counts_filtered = df[df['NER'] != 0]['NER'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
ner_counts_filtered.plot(kind='bar', color='green')
plt.title('Count of Non-zero NER Values')
plt.xlabel('NER ID Value')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#validationdata
ner_counts_filtered = df_val[df_val['NER'] != 0]['NER'].value_counts()
# Plotting
plt.figure(figsize=(10, 6))
ner_counts_filtered.plot(kind='bar', color='blue')
plt.title('Count of Non-zero NER Values')
plt.xlabel('NER ID Value')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#testingndata
ner_counts_filtered = df_test[df_test['NER'] != 0]['NER'].value_counts()
# Plotting
plt.figure(figsize=(10, 6))
ner_counts_filtered.plot(kind='bar', color='red')
plt.title('Count of Non-zero NER Values')
plt.xlabel('NER ID Value')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#trainingdata
word_counts = df.groupby("SentenceID")["Word"].agg(["count"])
word_counts = word_counts.rename(columns={"count": "Word count"})
word_counts.hist(bins=50, figsize=(8,6));

#validationdata
word_counts_val = df_val.groupby("SentenceID")["Word"].agg(["count"])
word_counts_val = word_counts_val.rename(columns={"count": "Word count"})
word_counts_val.hist(bins=50, figsize=(8,6));

#testingdata
word_counts_test = df_test.groupby("SentenceID")["Word"].agg(["count"])
word_counts_test = word_counts_test.rename(columns={"count": "Word count"})
word_counts_test.hist(bins=50, figsize=(8,6));

#trainingdata
MAX_SENTENCE = word_counts.max()[0]
print("Longest sentence in the training corpus contains {} words.".format(MAX_SENTENCE))
longest_sentence_id = word_counts[word_counts["Word count"]==MAX_SENTENCE].index[0]
print("ID of the longest sentence in training set is {}.".format(longest_sentence_id))
longest_sentence = df[df["SentenceID"]==longest_sentence_id]["Word"].str.cat(sep=' ')
print("The longest sentence in the training corpus is:",longest_sentence)

#validationdata
MAX_SENTENCE_VAL = word_counts_val.max()[0]
print("Longest sentence in the validation corpus contains {} words.".format(MAX_SENTENCE_VAL))
longest_sentence_id_val = word_counts_val[word_counts_val["Word count"]==MAX_SENTENCE_VAL].index[0]
print("ID of the longest sentence in validation set is {}.".format(longest_sentence_id_val))
longest_sentence_val = df_val[df_val["SentenceID"]==longest_sentence_id_val]["Word"].str.cat(sep=' ')
print("The longest sentence in the validation corpus is:",longest_sentence_val)

##testingdata
MAX_SENTENCE_TEST = word_counts_test.max()[0]
print("Longest sentence in the testing corpus contains {} words.".format(MAX_SENTENCE_TEST))
longest_sentence_id_test = word_counts_test[word_counts_test["Word count"]==MAX_SENTENCE_TEST].index[0]
print("ID of the longest sentence in testing set is {}.".format(longest_sentence_id_test))
longest_sentence_test = df_test[df_test["SentenceID"]==longest_sentence_id_test]["Word"].str.cat(sep=' ')
print("The longest sentence in the testing corpus is:",longest_sentence_test)

#trainingdata
all_words = list(set(df["Word"].values))
all_tags = list(set(df["NER"].values))
print("Number of unique words in training set: {}".format(df["Word"].nunique()))
print("Number of unique tags in training set : {}".format(df["NER"].nunique()))

#validationdata
all_words_val = list(set(df_val["Word"].values))
all_tags_val = list(set(df_val["NER"].values))
print("Number of unique words in validation set: {}".format(df_val["Word"].nunique()))
print("Number of unique tags in validation set : {}".format(df_val["NER"].nunique()))

#testingdata
all_words_test = list(set(df_test["Word"].values))
all_tags_test = list(set(df_test["NER"].values))
print("Number of unique words in validation set: {}".format(df_test["Word"].nunique()))
print("Number of unique tags in validation set : {}".format(df_test["NER"].nunique()))

#training
word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
word2index["--UNKNOWN_WORD--"]=0
word2index["--PADDING--"]=1
index2word = {idx: word for word, idx in word2index.items()}

#validation
word2index_val = {word: idx + 2 for idx, word in enumerate(all_words_val)}
word2index_val["--UNKNOWN_WORD--"]=0
word2index_val["--PADDING--"]=1
index2word_val = {idx: word for word, idx in word2index_val.items()}

#testing
word2index_test = {word: idx + 2 for idx, word in enumerate(all_words_test)}
word2index_test["--UNKNOWN_WORD--"]=0
word2index_test["--PADDING--"]=1
index2word_test = {idx: word for word, idx in word2index_test.items()}

for k,v in sorted(word2index.items(), key=operator.itemgetter(1))[:10]:
    print(k,v)
for k,v in sorted(word2index_val.items(), key=operator.itemgetter(1))[:10]:
    print(k,v)
for k,v in sorted(word2index_test.items(), key=operator.itemgetter(1))[:10]:
    print(k,v)

test_word = "impressive"
test_word_idx = word2index[test_word]
test_word_lookup = index2word[test_word_idx]

print("The index of the word {} is {}.".format(test_word, test_word_idx))
print("The word with index {} is {}.".format(test_word_idx, test_word_lookup))

tag2index = {tag: idx for idx, tag in enumerate(all_tags)}
tag2index["--PADDING--"] = 37
index2tag = {idx: word for word, idx in tag2index.items()}

#validation
tag2index_val = {tag: idx for idx, tag in enumerate(all_tags_val)}
tag2index_val["--PADDING--"] = 37
index2tag_val= {idx: word for word, idx in tag2index_val.items()}

#testing
tag2index_test = {tag: idx for idx, tag in enumerate(all_tags_test)}
tag2index_test["--PADDING--"] = 37
index2tag_test = {idx: word for word, idx in tag2index_test.items()}

def to_tuples(data):
    iterator = zip(data["Word"].values.tolist(), data["NER"].values.tolist())
    return [(word,ner) for word, ner in iterator]

sentences = df.groupby("SentenceID").apply(to_tuples).tolist()
sentences_val = df_val.groupby("SentenceID").apply(to_tuples).tolist()
sentences_test = df_test.groupby("SentenceID").apply(to_tuples).tolist()

print(sentences[0])
print(sentences_val[0])
print(sentences_test[0])

#training
X = [[word[0] for word in sentence] for sentence in sentences]
y = [[word[1] for word in sentence] for sentence in sentences]
print("X[0]:", X[5])
print("y[0]:", y[5])

X_index = [[word2index[word] for word in sentence] for sentence in X]
y_index = [[tag2index[tag] for tag in sentence] for sentence in y]
print("X[0]:", X_index[5])
print("y[0]:", y_index[5])

X_padded = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X_index]
y_padded = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y_index]
print("X[0]:", X_padded[5])
print("y[0]:", y_padded[5])

X_tensor = torch.tensor(X_padded)
y_tensor = torch.tensor(y_padded)

print("X_tensor[0]:", X_tensor[0])
print("y_tensor[0]:", y_tensor[0])

#validation
X_val = [[word[0] for word in sentence] for sentence in sentences_val]
y_val = [[word[1] for word in sentence] for sentence in sentences_val]
print("X[0]:", X_val[0])
print("y[0]:", y_val[0])

X_index_val = [[word2index_val[word] for word in sentence] for sentence in X_val]
y_index_val = [[tag2index_val[tag] for tag in sentence] for sentence in y_val]
print("X[0]:", X_index_val[0])
print("y[0]:", y_index_val[0])

X_padded_val = [sentence + [word2index_val["--PADDING--"]] * (MAX_SENTENCE_VAL - len(sentence)) for sentence in X_index_val]
y_padded_val = [sentence + [tag2index_val["--PADDING--"]] * (MAX_SENTENCE_VAL - len(sentence)) for sentence in y_index_val]
print("X[0]:", X_padded_val[0])
print("y[0]:", y_padded_val[0])

X_tensor_val = torch.tensor(X_padded_val)
y_tensor_val = torch.tensor(y_padded_val)

print("X_tensor[0]:", X_tensor_val[0])
print("y_tensor[0]:", y_tensor_val[0])

#testing
X_test = [[word[0] for word in sentence] for sentence in sentences_test]
y_test = [[word[1] for word in sentence] for sentence in sentences_test]
print("X[0]:", X_test[0])
print("y[0]:", y_test[0])

X_index_test = [[word2index_test[word] for word in sentence] for sentence in X_test]
y_index_test = [[tag2index_test[tag] for tag in sentence] for sentence in y_test]
print("X[0]:", X_index_test[0])
print("y[0]:", y_index_test[0])

X_padded_test = [sentence + [word2index_test["--PADDING--"]] * (MAX_SENTENCE_TEST - len(sentence)) for sentence in X_index_test]
y_padded_test = [sentence + [tag2index_test["--PADDING--"]] * (MAX_SENTENCE_TEST - len(sentence)) for sentence in y_index_test]
print("X[0]:", X_padded_test[0])
print("y[0]:", y_padded_test[0])

X_tensor_test = torch.tensor(X_padded_test)
y_tensor_test = torch.tensor(y_padded_test)

print("X_tensor[0]:", X_tensor_test[0])
print("y_tensor[0]:", y_tensor_test[0])

len(y_tensor_test)

'''from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

print("Number of sentences in the training dataset: {}".format(len(X_train)))
print("Number of sentences in the test dataset : {}".format(len(X_test)))'''

X_train=X_tensor
y_train=y_tensor
X_test=X_tensor_test
y_test=y_tensor_test

print("Number of sentences in the training dataset: {}".format(len(X_train)))
print("Number of sentences in the test dataset : {}".format(len(X_test)))
print("Number of ner in the training dataset: {}".format(len(y_train)))
print("Number of ner in the test dataset : {}".format(len(y_test)))

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

print("Number of sentences in the training dataset: {}".format(len(X_train)))
print("Number of sentences in the test dataset : {}".format(len(X_test)))

# Define model hyperparameters
WORD_COUNT = len(index2word)
DENSE_EMBEDDING = 100
LSTM_UNITS = 100
LSTM_DROPOUT = 0.1
DENSE_UNITS = 200
TAG_COUNT = len(index2tag)
BATCH_SIZE = 256
MAX_EPOCHS = 20

# Step 2: Define Model Architecture
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        linear_out = self.linear(lstm_out)
        return linear_out

# Step 3: Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(WORD_COUNT, DENSE_EMBEDDING, LSTM_UNITS, TAG_COUNT).to(device)

# Step 4: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train Model
for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0.0
    for i in range(0, len(X_train), BATCH_SIZE):
        inputs = torch.tensor(X_train[i:i+BATCH_SIZE], device=device)
        targets = torch.tensor(y_train[i:i+BATCH_SIZE], device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, TAG_COUNT)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{MAX_EPOCHS}, Loss: {total_loss}")

'''#evaluation
model.eval()
total_correct = 0
total_samples = 0
predicted_all = []
targets_all = []

with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE):
        inputs = torch.tensor(X_test[i:i+BATCH_SIZE], device=device) if torch.cuda.is_available() else torch.tensor(X_test[i:i+BATCH_SIZE])
        targets = torch.tensor(y_test[i:i+BATCH_SIZE], device=device) if torch.cuda.is_available() else torch.tensor(y_test[i:i+BATCH_SIZE])

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 2)
        #print(_,predicted)
        #print(_.shape)
        #print(predicted.shape)

        mask = (targets != 37)
        total_correct += torch.sum((predicted == targets)[mask]).item()
        total_samples += torch.sum(mask).item()

        total_correct += torch.sum(predicted == targets).item()
        total_samples += targets.numel()

        #predicted_all.extend(predicted.cpu().flatten().tolist())  # Move to CPU if needed
        #targets_all.extend(targets.cpu().flatten().tolist())  # Move to CPU if needed

        predicted_all.extend(predicted[mask].flatten().tolist())  # Move to CPU if needed
        targets_all.extend(targets[mask].flatten().tolist())

accuracy = total_correct / total_samples
f1 = f1_score(targets_all, predicted_all, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)'''

model.eval()

# Initialize lists to store metrics for each class
class_precision = []
class_recall = []
class_f1 = []
with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE):
        inputs = torch.tensor(X_test[i:i+BATCH_SIZE], device=device) if torch.cuda.is_available() else torch.tensor(X_test[i:i+BATCH_SIZE])
        targets = torch.tensor(y_test[i:i+BATCH_SIZE], device=device) if torch.cuda.is_available() else torch.tensor(y_test[i:i+BATCH_SIZE])

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 2)

        predicted_np = predicted.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(targets_np, predicted_np, average=None)

        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)

max_precision_shape = max([len(p) for p in class_precision])
max_recall_shape = max([len(r) for r in class_recall])
max_f1_shape = max([len(f) for f in class_f1])
max_shape = max(max_precision_shape, max_recall_shape, max_f1_shape)

class_precision = [np.pad(precision, (0, max_shape - len(precision)), mode='constant') for precision in class_precision]
class_recall = [np.pad(recall, (0, max_shape - len(recall)), mode='constant') for recall in class_recall]
class_f1 = [np.pad(f1, (0, max_shape - len(f1)), mode='constant') for f1 in class_f1]

class_precision = np.array(class_precision)
class_recall = np.array(class_recall)
class_f1 = np.array(class_f1)

avg_precision = np.mean(class_precision, axis=0)
avg_recall = np.mean(class_recall, axis=0)
avg_f1 = np.mean(class_f1, axis=0)

avg_precision_all = np.mean(class_precision)
avg_recall_all = np.mean(class_recall)
avg_f1_all = np.mean(class_f1)

# Print average metrics for all classes
print("Average Precision (All Classes):", avg_precision_all)
print("Average Recall (All Classes):", avg_recall_all)
print("Average F1-score (All Classes):", avg_f1_all)

# Print metrics for each class
for c in range(len(avg_precision)):
    print(f"Class {c}: Precision={avg_precision[c]}, Recall={avg_recall[c]}, F1-score={avg_f1[c]}")

#torch.save(model.state_dict(), '/content/drive/My Drive/NLPmodel.pth')

#below is the non-weighted class-wise parameters result
'''
Average Precision (All Classes): 0.6518060278487932
Average Recall (All Classes): 0.6396354672750366
Average F1-score (All Classes): 0.6291673633651345
Class 0: Precision=0.987457421531131, Recall=0.9852142638915139, F1-score=0.9863261926901565
Class 1: Precision=0.8485482350781257, Recall=0.837485390743533, F1-score=0.841686858029688
Class 2: Precision=0.8784016301922574, Recall=0.8023192889991965, F1-score=0.8357710755326787
Class 3: Precision=0.8080889304871097, Recall=0.8944401522559169, F1-score=0.8440496906723495
Class 4: Precision=0.6630434782608695, Recall=0.5905797101449275, F1-score=0.6016218081435474
Class 5: Precision=0.4284161490683229, Recall=0.3774638405073187, F1-score=0.3734150511324425
Class 6: Precision=0.5575086690653249, Recall=0.5573569451141152, F1-score=0.5384781478840192
Class 7: Precision=0.8074608077846714, Recall=0.8426163320385102, F1-score=0.8225446799538964
Class 8: Precision=0.8457715978446451, Recall=0.8685351093160634, F1-score=0.8547477257357879
Class 9: Precision=0.8271072148579905, Recall=0.8700636116212148, F1-score=0.8435008889887253
Class 10: Precision=0.751565560361327, Recall=0.7905259914850707, F1-score=0.7490473433497608
Class 11: Precision=0.5976356373095504, Recall=0.6185300207039337, F1-score=0.5832133849060759
Class 12: Precision=0.5801166118621363, Recall=0.5956874647092036, F1-score=0.5613258830849788
Class 13: Precision=0.5679520427062961, Recall=0.55852430307596, F1-score=0.5266171180684228
Class 14: Precision=0.6989408339249752, Recall=0.6505718984365703, F1-score=0.6616852578962332
Class 15: Precision=0.7719914788436205, Recall=0.8340494316141635, F1-score=0.7920587675944637
Class 16: Precision=0.6399852437276972, Recall=0.6677596552999839, F1-score=0.6336402332317773
Class 17: Precision=0.5173982720178374, Recall=0.5281832298136647, F1-score=0.5045502784911573
Class 18: Precision=0.6983715559802518, Recall=0.7350196142530239, F1-score=0.7061780710316927
Class 19: Precision=0.9094651158236724, Recall=0.9065415163338912, F1-score=0.9026020888299547
Class 20: Precision=0.9218091545888164, Recall=0.8774898358464744, F1-score=0.8925352209482078
Class 21: Precision=0.7889663807319512, Recall=0.8290269168685852, F1-score=0.7954105634394508
Class 22: Precision=0.6630720220799625, Recall=0.6828097377841624, F1-score=0.6512458605329028
Class 23: Precision=0.6159220219961908, Recall=0.6055200320461459, F1-score=0.5894202844241764
Class 24: Precision=0.6233992229623219, Recall=0.6488849505439895, F1-score=0.6184854955540839
Class 25: Precision=0.69454034125126, Recall=0.7107586389163948, F1-score=0.6829181043110697
Class 26: Precision=0.7063477399085211, Recall=0.710247104052765, F1-score=0.6878881150040357
Class 27: Precision=0.649750822848649, Recall=0.5565496098104794, F1-score=0.5838517749949536
Class 28: Precision=0.5681159420289855, Recall=0.49046756383712914, F1-score=0.48286098277517037
Class 29: Precision=0.43959627329192547, Recall=0.34389418481490086, F1-score=0.3599946817590903
Class 30: Precision=0.6239130434782609, Recall=0.4143461007591442, F1-score=0.4657789070832549
Class 31: Precision=0.5539227680532028, Recall=0.48227835769256877, F1-score=0.4994600818252896
Class 32: Precision=0.4892339544513458, Recall=0.44515810276679846, F1-score=0.45102169412188314
Class 33: Precision=0.38250517598343686, Recall=0.3752242926155969, F1-score=0.3771797256451988
Class 34: Precision=0.22826086956521738, Recall=0.21231884057971012, F1-score=0.21847826086956523
Class 35: Precision=0.13043478260869565, Recall=0.13043478260869565, F1-score=0.13043478260869565
'''