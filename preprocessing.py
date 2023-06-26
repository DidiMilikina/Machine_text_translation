from __future__ import unicode_literals, print_function, division
from io import open
import torch
import csv
from transformers import BertTokenizer, BertModel, BertConfig

from hyperparameters import *

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def get_device():
    """
    Checks if a GPU is available
    Returns:
        available device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device

def read_input(lang1, lang2):
    """
    Read pre-specified NUM_LINES_READ from the dataset.
    Args:
        lang1, lang2
    Returns:
        lang1, lang2, data
    """
    data = []
    with open('en-fr.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, skipinitialspace=True, delimiter=',')

        for _ in range(NUM_LINES_READ):
            line = next(reader)
            # Skip column names
            if line == ['en', 'fr']:
                continue  # Skip the line           

            data.append(line)

    return lang1, lang2, data


def encode_data(dataset, en_tokenizer, fr_tokenizer):
    """
    Encodes English and French sentences.
    Args:
        dataset, en_tokenizer, fr_tokenizer
    Returns:
        encoded_sequences: structure(int tokens): [ [[en],[fr]], [[en],[fr]] ]
    """
    encoded_sequences = []
    for seq in dataset:
        seq_en = en_tokenizer.encode(seq[0], truncation=True)
        seq_fr = fr_tokenizer.encode(seq[1], truncation=True)
        
        encoded_sequences.append([seq_en, seq_fr])

    return encoded_sequences


def dataset_split_sorted(data):
    """
    Splits the initial dataset into train, val, test datasets based on the predefined ratios per batch.
    Sorts the data in each dataset based on the longest English sequence.
    If the batches in each dataset are not of length BATCH_SIZE a value error will be raised.
    Args: 
        data: entire dataset
    Returns:
        train_pairs_sorted, val_pairs_sorted, test_pairs_sorted
        OR
        raises value error
    """
    total_size = len(data)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    train_pairs = data[:train_size]
    train_pairs_sorted = sorted(train_pairs, key=lambda x: len(x[0]))

    val_pairs = data[train_size:train_size + val_size]
    val_pairs_sorted = sorted(val_pairs, key=lambda x: len(x[0]))

    test_pairs = data[train_size + val_size:]
    test_pairs_sorted = sorted(test_pairs, key=lambda x: len(x[0]))

    # Check if the last batch in each set has size equal to BATCH_SIZE
    train_last_batch_size = train_size % BATCH_SIZE
    val_last_batch_size = val_size % BATCH_SIZE
    test_last_batch_size = test_size % BATCH_SIZE

    if train_last_batch_size != 0:
        missing_train_lines = BATCH_SIZE - train_last_batch_size
        raise ValueError(f"Last batch in train set has size {train_last_batch_size}, expected {BATCH_SIZE}. Missing lines: {missing_train_lines}")
    if val_last_batch_size != 0:
        missing_val_lines = BATCH_SIZE - val_last_batch_size
        raise ValueError(f"Last batch in validation set has size {val_last_batch_size}, expected {BATCH_SIZE}. Missing lines: {missing_val_lines}")
    
    if test_last_batch_size != 0:
        missing_test_lines = BATCH_SIZE - test_last_batch_size
        raise ValueError(f"Last batch in test set has size {test_last_batch_size}, expected {BATCH_SIZE}. Missing lines: {missing_test_lines}")
    
    return train_pairs_sorted, val_pairs_sorted, test_pairs_sorted


def data_prep(lang1, lang2, bert_en_tokenizer, bert_fr_tokenizer):
    """
    Helper function.
    Calls functions read_input, encode_data
    Args:
        lang1, lang2, bert_en_tokenizer, bert_fr_tokenizer
    Returns:
        input_lang, output_lang, train_data_sorted, val_data_sorted, test_data_sorted
        OR 
        raises value error
    """
    try:
        input_lang, output_lang, data = read_input(lang1, lang2)    
        encoded_dataset = encode_data(data, bert_en_tokenizer, bert_fr_tokenizer)
        train_data_sorted, val_data_sorted, test_data_sorted = dataset_split_sorted(encoded_dataset)

        return input_lang, output_lang, train_data_sorted, val_data_sorted, test_data_sorted

    except ValueError as e:
        print("Error:", str(e))
        return None
    

def add_padding(sorted_sliced_batch): 
    """
    Adds padding to all sequences with len lower than the len of the longest sequence in a sorted batch from both languages.
    If the sequence has the same len as the longest sequencey, no padding will be added.
    The longest seqeunce is the last element of the sorted batch.
    The batch is sorted based on the length of the English sentencs.
    Args:   
        sorted_sliced_batch: batch/dataset containing tuples([article],[summary])

    Returns:
        [(tensor([[article],[article]]),tensor([[summary],[summary]]))]: a list of lists of torch tensor for articles per batch and torch tensor for summaries per batch
    """

    max_len = 0
    current_len = 0
    out_en = []
    out_fr = []

    for k in sorted_sliced_batch:
        for i in k:
            for j in i:
                current_len = len(j)
                if current_len > max_len:
                    max_len = current_len
        
        # Apply padding to all lists in the structure
        padded_en = []
        padded_fr = []

        for q in k:
            for index, w in enumerate(q):
                if index == 0: 
                    # Pad the current list with zeros
                    padding_length = max_len - len(w)
                    padded_list = w + [0] * padding_length
                    padded_en.append(padded_list)

                if index == 1:
                    # Pad the current list with zeros
                    padding_length = max_len - len(w)
                    padded_list = w + [0] * padding_length
                    padded_fr.append(padded_list)
            
        out_en.append(torch.tensor(padded_en))
        out_fr.append(torch.tensor(padded_fr))
        current_len = 0
        max_len = 0   
    
    return out_en, out_fr

def slice_batches_fixed_sequences(dataset, el_in_list):  
    """
    Slices the input dataset into batches containing a fixed number of sequences of reviews per batch. 
    A fixed number of sequences per batch. 

    Args:
        dataset: list of tuples(sequence, sentiment)
        el_in_list (int): number of sequences per batch
    
    Returns: 
        dataset of tuples(sequence, sentiment) up to the specified number of reviews.
    """

    out = []
    for i in range(0, len(dataset), el_in_list):
        out.append(dataset[i:i+el_in_list])
    return out 


# Initialize the BERT tokenizer
# For English
bert_en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# For French
bert_fr_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

bert_en_config = BertConfig().from_pretrained('bert-base-uncased')
bert_en_model = BertModel(bert_en_config)
# Take the vocab size
BERT_EN_VOCAB_SIZE = bert_en_model.config.vocab_size

bert_en_config = BertConfig().from_pretrained('bert-base-multilingual-uncased')
bert_en_model = BertModel(bert_en_config)
# Take the vocab size
BERT_FR_VOCAB_SIZE = bert_en_model.config.vocab_size

input_lang, output_lang, train_data_sorted, val_data_sorted, test_data_sorted = data_prep('eng', 'fra', bert_en_tokenizer, bert_fr_tokenizer)
sliced_batch_train = slice_batches_fixed_sequences(dataset=train_data_sorted, el_in_list=BATCH_SIZE)
pad_train_en, pad_train_fr = add_padding(sliced_batch_train)

sliced_batch_val = slice_batches_fixed_sequences(dataset=val_data_sorted, el_in_list=BATCH_SIZE)
pad_val_en, pad_val_fr = add_padding(sliced_batch_val)

sliced_batch_test = slice_batches_fixed_sequences(dataset=test_data_sorted, el_in_list=BATCH_SIZE)
pad_test_en, pad_test_fr = add_padding(sliced_batch_test)