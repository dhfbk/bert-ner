# bert-ner

A script to train a named-entities recognizer with [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)).
It works out-of-the-box for dataset having three labels: PER, LOC, ORG.

It is partially inspired by [this blogpost](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/) by Tobias Sterbak.

```
usage: train-bert.py [-h] [--bert BERT] [--max_len MAX_LEN] [--epochs EPOCHS] {train,test} file model

Train a NER model with BERT. Labels must be PER, ORG, LOC.

positional arguments:
  {train,test}       Choose whether to train the model or test it
  file               TSV file containing train/test data
  model              Model folder (it must exists in 'test' action)

optional arguments:
  -h, --help         show this help message and exit
  --bert BERT        BERT model name in Huggingface (default dbmdz/bert-base-italian-cased)
  --max_len MAX_LEN  BERT maximum length of sequence (default 200)
  --epochs EPOCHS    Number of training epochs (default 3)
```
