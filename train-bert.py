import argparse

defaultBert = 'dbmdz/bert-base-italian-cased'
defaultMaxLen = 100

parser = argparse.ArgumentParser(description='Train a NER model with BERT. Labels must be PER, ORG, LOC.')
parser.add_argument('action', choices=['train', 'test'], help="Choose whether to train the model or test it")
parser.add_argument('file', help="TSV file containing train/test data")
parser.add_argument('model', help="Model folder (it must exists in 'test' action)")
parser.add_argument('--bert', help="BERT model name in Huggingface (default " + defaultBert + ")", required=False, default=defaultBert)
parser.add_argument('--max_len', help="BERT maximum length of sequence (default " + str(defaultMaxLen) + ")", required=False, default=defaultMaxLen)
args = parser.parse_args()

import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm, trange
import numpy as np
import os
import re

ner_filename = args.file
modelFolder = args.model
bert_model = args.bert
maxLen = args.max_len
train = True
if args.action == "test":
    train = False

###

bs = 32
tag_values = ["ORG", "O", "LOC", "PER"]
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

ne_re = re.compile(r"^(.*)\s([^\s]+)$")

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

if train:
    print("### TRAINING MODEL")
    tokens = []
    labels = []
    tokenized_texts_and_labels = []
    with open(ner_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                tokenized_bert = tokenize_and_preserve_labels(tokens, labels)
                tokenized_texts_and_labels.append(tokenized_bert)
                tokens = []
                labels = []
                continue
            m = ne_re.match(line)
            tokens.append(m.group(1))
            labels.append(m.group(2))

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=maxLen, dtype="long", value=0.0,
        truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
        maxlen=maxLen, value=tag2idx["PAD"], padding="post",
        dtype="long", truncating="post")

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    model = BertForTokenClassification.from_pretrained(
        bert_model,
        num_labels = len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda();

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    epochs = 3
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        # print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

    os.makedirs(modelFolder)
    model.save_pretrained(modelFolder)

else:
    print("### TESTING MODEL")
    tokens = []
    labels = []
    tokenized_texts_and_labels = []
    with open(ner_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                tokenized_bert = tokenize_and_preserve_labels(tokens, labels)
                tokenized_texts_and_labels.append(tokenized_bert)
                tokens = []
                labels = []
                continue
            m = ne_re.match(line)
            tokens.append(m.group(1))
            labels.append(m.group(2))

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    all_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    tags = [[tag2idx.get(l) for l in lab] for lab in labels]

    model = BertForTokenClassification.from_pretrained(modelFolder)
    model.cuda();

    all_labels = []
    all_gold = []
    for i in range(len(all_ids)):

        input_ids = torch.tensor([all_ids[i]]).cuda()

        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

        new_tokens, new_labels, gold_labels = [], [], []
        for token, label_idx, gold_label in zip(tokens, label_indices[0], tags[i]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
                gold_labels.append(tag_values[gold_label])

        all_labels += new_labels
        all_gold += gold_labels

    print("Macro:", precision_recall_fscore_support(all_gold, all_labels, average='macro', labels=["PER", "LOC", "ORG"])[:3])
    print("Micro:", precision_recall_fscore_support(all_gold, all_labels, average='micro', labels=["PER", "LOC", "ORG"])[:3])
    results = precision_recall_fscore_support(all_gold, all_labels, average=None, labels=["PER", "LOC", "ORG"])
    support = results[3]
    results = np.delete(results, 3, axis=0)
    results = np.transpose(results)
    print("PER:", results[0])
    print("LOC:", results[1])
    print("ORG:", results[2])
    print("Support:", support)
