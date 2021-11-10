# bert-ner

A script to train a named-entities recognizer with [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)).

Data must be in the TSV format `word [tab] label` with newline as sentence separator.

It is partially inspired by [this blogpost](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/) by Tobias Sterbak.

```
usage: train-bert.py [-h] [-b, --bert NAME] [-l, --max_len LEN] [-e, --epochs NUM] [-p, --pad_label LBL] {train,test} file model

Train a NER model with BERT.

positional arguments:
  {train,test}         Choose whether to train the model or test it
  file                 TSV file containing train/test data
  model                Model folder (it must exists in 'test' action)

optional arguments:
  -h, --help           show this help message and exit
  -b, --bert NAME      BERT model name in Huggingface (default dbmdz/bert-base-italian-cased)
  -l, --max_len LEN    BERT maximum length of sequence (default 200)
  -e, --epochs NUM     Number of training epochs (default 3)
  -p, --pad_label LBL  Padding label, it must be different from labels in the NER data (default PAD)
```
