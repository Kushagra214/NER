# Team Name - Simplexity
# Team Members - Kushagra Agarwal, Sarthak Parakh
# Description -

# In this code, We are implementing the Active Learning methodology to train the model on the subset of training data \
# and implement uncertainty sampling on remaining set of training data to find the samples which the model is most uncertain about \
# and used those labeled samples to add to the training set and retrain until good accuracy is achieved.

# NLP concepts used - 
# I. Syntax - Named Entity Recognition modeling
# III. Language Modeling|Transformers - RoBERTA model
# IV. Applications - Active Learning for unlabeled datasets

# In this code, the optimal solution hasn't been achieved yet, currently working on it.

# Ran the code on Google Colab leveraging T4 GPU

# DatasetDict({
#     labeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 8019
#     })
#     unlabeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 32077
#     })
# })




import os, sys, random, re, collections, string
import numpy as np
import math
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
import torch.nn as nn
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Features
from transformers import RobertaModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_metric



# Ontonotes V5 english dataset
dataset = load_dataset("conll2012_ontonotesv5",'english_v4')

# roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', add_special_tokens=False)
# RoBERTa model
model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=37)
# to help with padding
data_collator = DataCollatorForTokenClassification(tokenizer)

label_names = dataset['train'].features['sentences'][0]['named_entities'].feature.names


# Define your Trainer class for NER
class NERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.CrossEntropyLoss()
        if return_outputs:
            return loss(logits, inputs["labels"]), outputs
        else:
            return loss(logits, inputs["labels"])

# Define your uncertainty sampling function
def uncertainty_sampling(dataset, model):
    scores = []
    for inputs in dataset:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs, dim=-1)[0]
            uncertainty_score = 1 - max_prob.item()  # Using 1 - max probability as uncertainty measure
            scores.append((uncertainty_score, inputs))
    scores.sort(reverse=True)  # Sort by uncertainty score in descending order
    return [inputs for _, inputs in scores]


# compute F-1 score of Multi-class classification
metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    #select predicted index with maximum logit for each token
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# adjust (duplicate) labels when the words get tokenised by the tokeniser
def tokenize_adjust_labels(x, y):
    tokenized_samples = tokenizer.batch_encode_plus(x, is_split_into_words=True, truncation=True, return_special_tokens_mask=False)
    labels = [-100]
    input = [0]
    attention = [1]

    # Iterate through each entry in input_ids
    for i, input_ids in enumerate(tokenized_samples['input_ids']):
        # Convert input_ids to tokens
        main_tokens = input_ids[1:-1]
        for j in range(len(main_tokens)):
          input.append(main_tokens[j])
          labels.append(y[i])
          attention.append(1)
        
    input.append(2)    
    labels.append(-100)    
    attention.append(1)

    data = {
        'input_ids': input,
        'attention_mask': attention,
        'labels': labels,
      }
    
    return data





def main():

    print("Dataset: \n")
    print(dataset)

    print("Named Entities in the dataset :\n")
    label_names = dataset['train'].features['sentences'][0]['named_entities'].feature.names
    print(label_names)
    print("\nNumber of Entities :\n")
    print(len(label_names))


    # Initialize a set to store unique substrings
    unique_substrings = set()
    # Iterate over each string
    for string in dataset['train']['document_id']:
        # Find the index of the first '/'
        index = string.find('/')
        if index != -1:  # If '/' is found
            # Extract substring till the first '/'
            substring = string[:index]
            # Add the substring to the set
            unique_substrings.add(substring)
        else:  # If '/' is not found
            # Add the entire string to the set
            unique_substrings.add(string)

    # Print unique substrings
    print("Docs in the dataset:", unique_substrings)


    print("Example of the dataset : \n")
    print('Document Id : ',dataset['train'][0]['document_id'])
    print('Sentence Dictionary :')
    print(dataset['train'][0]['sentences'][0])


    print("Train set: 3 broad categories as a news category with not much linguistic variation: \
        broadcast news (bn), magazines (mz), newswire (nw). \n \
        Test Set: our model’s performance on the 3 seen categories as well as 3 unseen categories: \
        broadcast conversation (bc), telephone conversation (tc), weblog (wb).")
    
    train_idx=[]
    for i, string in enumerate(dataset['train']['document_id']):
        # Find the index of the first '/'
        index = string.find('/')
        if index != -1:  # If '/' is found
            # Extract substring till the first '/'
            substring = string[:index]
            # print(i)
            # print(type(substring))
            if substring in ["bn", "mz","nw"]:
                train_idx.append(i)

    test_idx=[]
    for i, string in enumerate(dataset['test']['document_id']):
        # Find the index of the first '/'
        index = string.find('/')
        if index != -1:  # If '/' is found
            # Extract substring till the first '/'
            substring = string[:index]
            # print(i)
            # print(type(substring))
            if substring in ["bn", "mz","nw"]:
                test_idx.append(i)


    val_idx=[]
    for i, string in enumerate(dataset['validation']['document_id']):
        # Find the index of the first '/'
        index = string.find('/')
        if index != -1:  # If '/' is found
            # Extract substring till the first '/'
            substring = string[:index]
            # print(i)
            # print(type(substring))
            if substring in ["bn", "mz","nw"]:
                val_idx.append(i)

    print("\nSentence: ", dataset['train'][3]['sentences']['words'])
    print("\nNER Labels: ",dataset['train'][3]['sentences']['named_entities'])

    # sentences
    X_train = []
    # ner
    y_train = []

    for i in train_idx:
        for data in dataset['train'][i]['sentences']:
            X_train.append(data['words'])
            y_train.append(data['named_entities'])

    print('Number of training samples before augmentation -', len(X_train))

    ct = 0
    for i in range(len(X_train)):
        if is_more_than_half_non_zero(y_train[i]):
            ct +=1
            input = ' '.join(X_train[i])
            augmented_sentence = augment_sentence_with_similar_ner_tag(input, y_train[i])
            if len(augmented_sentence.split(' ')) != len(y_train[i]):
                print("Yes")
                X_train.append(augmented_sentence.split(' '))
                y_train.append(y_train[i])

    print('Number of augmented training samples having 50% NER tags other than "O" added to the training set -', ct)

    print('Number of training samples after first augmentation -', len(X_train))

    # Calculate the number of indices to pick (20% of X_train length)
    num_indices_to_pick = int(len(X_train) * 0.20)

    # Randomly select indices without replacement
    random_indices = random.sample(range(len(X_train)), num_indices_to_pick)

    for i in random_indices:
        input = ' '.join(X_train[i])
        augmented_sentence = augment_sentence_with_similar_words(input)
        if len(augmented_sentence.split(' ')) == len(y_train[i]):
            # print("Yes")
            X_train.append(augmented_sentence.split(' '))
            y_train.append(y_train[i])

    print('Augmenting by replacing with similar words, 20% = ', len(random_indices), ' training samples random chosen')

    print('Number of training samples after second augmentation -', len(X_train))
    
    X_test=[]
    y_test=[]

    for i in test_idx:
        for data in dataset['test'][i]['sentences']:
            X_test.append(data['words'])
            y_test.append(data['named_entities'])

    X_val=[]
    y_val=[]

    for i in val_idx:
        for data in dataset['validation'][i]['sentences']:
            X_val.append(data['words'])
            y_val.append(data['named_entities'])

    # # List of input IDs
    train_input_ids=[]
    # # List of attention masks
    train_attention_masks = []
    # # List of labels
    train_labels = [] 

    for i in range(len(X_train)):
        t = tokenize_adjust_labels(X_train[i], y_train[i])
        train_input_ids.append(t['input_ids'])
        train_attention_masks.append(t['attention_mask'])
        train_labels.append(t['labels'])

    # Combine input IDs, attention masks, and labels into a dictionary
    data_train = {
        'input_ids': train_input_ids,
        'attention_mask': train_attention_masks,
        'labels': train_labels,
    }

    # # List of input IDs
    test_input_ids=[]
    # # List of attention masks
    test_attention_masks = []
    # # List of labels
    test_labels = [] 

    for i in range(len(X_test)):
        t = tokenize_adjust_labels(X_test[i], y_test[i])
        test_input_ids.append(t['input_ids'])
        test_attention_masks.append(t['attention_mask'])
        test_labels.append(t['labels'])

    # Combine input IDs, attention masks, and labels into a dictionary
    data_test = {
        'input_ids': test_input_ids,
        'attention_mask': test_attention_masks,
        'labels': test_labels,
    }


    # # List of input IDs
    val_input_ids=[]
    # # List of attention masks
    val_attention_masks = []
    # # List of labels
    val_labels = [] 

    for i in range(len(X_val)):
        t = tokenize_adjust_labels(X_val[i], y_val[i])
        val_input_ids.append(t['input_ids'])
        val_attention_masks.append(t['attention_mask'])
        val_labels.append(t['labels'])

    # Combine input IDs, attention masks, and labels into a dictionary
    data_val = {
        'input_ids': val_input_ids,
        'attention_mask': val_attention_masks,
        'labels': val_labels,
    }

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(data_train),
        'test': Dataset.from_dict(data_test),
        'val': Dataset.from_dict(data_val)
    })

    print(dataset_dict)

    # DatasetDict({
    #     train: Dataset({
    #         features: ['input_ids', 'attention_mask', 'labels'],
    #         num_rows: 40096
    #     })
    #     test: Dataset({
    #         features: ['input_ids', 'attention_mask', 'labels'],
    #         num_rows: 3930
    #     })
    #     val: Dataset({
    #         features: ['input_ids', 'attention_mask', 'labels'],
    #         num_rows: 3868
    #     })
    # })

    # Define your RoBERTa model
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=37)  # Adjust num_labels based on your proble

    batch_size = 16
    logging_steps = len(dataset_dict['train']) // batch_size
    epochs = 2


    #Create a subset of data_set for active learning
    # Calculate the number of samples for the subset (20%)
    subset_size = int(len(dataset_dict["train"]) * 0.2)

    # Create a DatasetDict
    train_subset_dataset_dict = DatasetDict({
        "labeled": dataset_dict["train"].select(range(subset_size)),
        "unlabeled": dataset_dict["train"].select(range(subset_size, len(dataset_dict["train"])))
    })

    print(train_subset_dataset_dict)

    # Define the TrainingArguments
    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        fp16=True,
        save_total_limit = 2,
        disable_tqdm=False,
        metric_for_best_model="f1",
    )

    # Initialize the Trainer with your NERTrainer class
    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset_dataset_dict["labeled"],
        eval_dataset=dataset_dict["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    

    # Fine-tune the model using uncertainty sampling
    for epoch in range(training_args.num_train_epochs):
        print(f"Epoch: {epoch}")
        
        print(train_subset_dataset_dict)

        # Get the 3000 most uncertain samples for annotation
        uncertain_samples = uncertainty_sampling(train_subset_dataset_dict["unlabeled"], model)
        
        # Add the uncertain samples to the training dataset
        train_subset_dataset_dict["labeled"] += uncertain_samples
        train_subset_dataset_dict["unlabeled"] -= uncertain_samples

        print(train_subset_dataset_dict)
        
        # Train the model for one epoch
        trainer.train()

    

    # setting up validation dataset for non-news data ["bc", "tc","wb"]
    diff_val_idx=[]
    for i, string in enumerate(dataset['validation']['document_id']):
        # Find the index of the first '/'
        index = string.find('/')
        if index != -1:  # If '/' is found
            # Extract substring till the first '/'
            substring = string[:index]
            # print(i)
            # print(type(substring))
            if substring in ["bc", "tc","wb"]:
                diff_val_idx.append(i)

    X_val_diff=[]
    y_val_diff=[]

    for i in diff_val_idx:
        for data in dataset['validation'][i]['sentences']:
            X_val.append(data['words'])
            y_val.append(data['named_entities'])

    # List of input IDs
    diff_val_input_ids=[]
    # List of attention masks
    diff_val_attention_masks = []
    # List of labels
    diff_val_labels = [] 

    for i in range(len(X_val_diff)):
        t = tokenize_adjust_labels(X_val_diff[i], y_val_diff[i])
        diff_val_input_ids.append(t['input_ids'])
        diff_val_attention_masks.append(t['attention_mask'])
        diff_val_labels.append(t['labels'])

    # Combine input IDs, attention masks, and labels into a dictionary
    data_val_diff = {
        'input_ids': diff_val_input_ids,
        'attention_mask': diff_val_attention_masks,
        'labels': diff_val_labels,
    }

    # Create a DatasetDict
    dataset_dict_diff = DatasetDict({
        'val_diff': Dataset.from_dict(data_val_diff)
    })

    print(dataset_dict_diff)

    # DatasetDict({
    #     val_diff: Dataset({
    #         features: ['input_ids', 'attention_mask', 'labels'],
    #         num_rows: 3868
    #     })
    # })

    # Set up evaluation
    evaluation_args = TrainingArguments(
        output_dir="results",
        per_device_eval_batch_size=batch_size,
        disable_tqdm=False,
    )

    # Create a new Trainer instance for evaluation
    eval_trainer = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=dataset_dict_diff['val_diff'],  # Use the different validation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #fine tune using train method
    eval_trainer.evaluate()




if __name__ == "__main__":
    main()





# Results-
    
# Epoch: 1

# DatasetDict({
#     labeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 8019
#     })
#     unlabeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 32077
#     })
# })
    
# Epoch: 2
    
# DatasetDict({
#     labeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 11019
#     })
#     unlabeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 29077
#     })
# })
    
# Epoch: 3
    
# DatasetDict({
#     labeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 14019
#     })
#     unlabeled: Dataset({
#         features: ['input_ids', 'attention_mask', 'labels'],
#         num_rows: 25077
#     })
# })


#   [7984/7984 18:53, Epoch 3/4]
#   Epoch Training Loss Validation Loss Precision Recall F1 Accuracy
#   1   3.971600 2.094341 0.479888 0.497074 0.588398 0.674937
#   2   2.287800 1.913614 0.492039 0.509071 0.590475 0.687053
#   3   1.941900 0.891489 0.506498 0.520556 0.613473 0.695210



# {'eval_loss': 2.09528690576553345,
# 'eval_precision': 0.7062342705112533,
# 'eval_recall': 0.6219458668617411,
# 'eval_f1': 0.6340225550277404,
# 'eval_accuracy': 0.77923486260093,
# 'eval_runtime': 11.0973,
# 'eval_samples_per_second': 348.553,
# 'eval_steps_per_second': 21.807}
    
